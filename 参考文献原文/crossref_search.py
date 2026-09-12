"""用 Crossref 反查研报挂错/存疑的 6 条引用真身。"""
import json
import subprocess
import sys
import time
import urllib.parse

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

QUERIES = [
    ("R3-F3", "Younsi", "wood high temperature heat treatment model"),
    ("R3-F7", "Champion", "glass transition diffusion water food"),
    ("R4-F18", "Takhar", "corn kernel stress moisture transport"),
    ("R4-F21", "Mercier", "sensitivity analysis drying model"),
    ("R4-F21", "Defraeye", "uncertainty convective drying porous materials"),
    ("R3-F4", "Ni Datta", "microwave drying heat mass transfer"),
]


def crossref(query, rows=4):
    url = ("https://api.crossref.org/works?rows=%d&select=DOI,title,author,"
           "container-title,issued&query.bibliographic=%s"
           % (rows, urllib.parse.quote(query)))
    r = subprocess.run(["curl", "-sSL", "--ssl-no-revoke", "-A", UA,
                        "--max-time", "40", url], capture_output=True, timeout=70)
    if r.returncode != 0 or not r.stdout.strip():
        return None
    try:
        return json.loads(r.stdout.decode("utf-8", errors="replace"))
    except Exception:
        return None


def main():
    out = []
    for key, author, phrase in QUERIES:
        print("=" * 92)
        print(f"[{key}]  研报称作者: {author}   主题: {phrase}")
        d = crossref(f"{author} {phrase}")
        if not d:
            print("   ❌ Crossref 无响应")
            out.append({"登记键": key, "查作者": author, "主题": phrase,
                        "候选": []})
            continue
        cands = []
        for it in d["message"]["items"]:
            au = ", ".join((x.get("family", "") + " " + x.get("given", "")[:1]).strip()
                           for x in (it.get("author") or [])[:3])
            yr = (it.get("issued", {}).get("date-parts") or [["?"]])[0][0]
            title = (it.get("title") or [""])[0]
            jrn = (it.get("container-title") or [""])[0]
            cands.append({"doi": it["DOI"], "title": title, "authors": au,
                          "journal": jrn, "year": yr})
            mark = "★" if author.split()[0].lower() in au.lower() else " "
            print(f"  {mark} {it['DOI']}")
            print(f"      {title[:86]}")
            print(f"      {au} | {jrn[:48]} | {yr}")
        out.append({"登记键": key, "查作者": author, "主题": phrase,
                    "候选": cands})
        time.sleep(0.6)
    print("=" * 92)
    with open("crossref_反查.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("→ crossref_反查.json")


if __name__ == "__main__":
    sys.exit(main())
