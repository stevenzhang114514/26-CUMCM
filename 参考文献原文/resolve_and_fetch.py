"""
文献解析与抓取（二版）              [甲 · 数据溯源审计配套工具]

一版 `fetch_refs.py` 暴露的问题
-------------------------------
1. 只查了 OpenAlex 的 `best_oa_location`，漏掉了其它 OA 副本
   → 改用 **OpenAlex 全部 locations + Semantic Scholar 的 openAccessPdf** 双源
2. 21 个 DOI 里 **6 个指向的论文与研报声称的完全不符** → 本版增加**标题反查**，
   用「作者 + 关键词」在 OpenAlex 里搜出**真正的那篇**

用法
----
    python 参考文献原文/resolve_and_fetch.py --check   # 只解析，不下载
    python 参考文献原文/resolve_and_fetch.py           # 解析 + 下载
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
PDF = BASE / "PDF"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# ---------------------------------------------------------------------------
# 研报里**挂错 DOI** 的 6 条：用作者+标题关键词反查真身
# (登记键, 研报声称的论文, 研报挂的错 DOI, 搜索用作者, 搜索用短语)
# ---------------------------------------------------------------------------
MISMATCH = [
    ("R3-F3", "Younsi et al. 2007, Int J Therm Sci（木材高温热处理三维模型）",
     "10.1016/j.ijthermalsci.2006.09.006", "Younsi", "high temperature heat treatment wood"),
    ("R3-F7", "Champion et al. 2000, J Food Eng（玻璃态中扩散系数跌落 10^2~10^4）",
     "10.1016/S0260-8774(00)00028-5", "Champion", "diffusion coefficient glassy"),
    ("R4-F18", "Takhar 2011, J Food Eng（玉米籽粒水分梯度致裂）",
     "10.1016/j.jfoodeng.2011.01.026", "Takhar", "corn kernel stress moisture"),
    ("R4-F21", "Mercier et al. 2013, J Food Eng（Sobol 灵敏度）",
     "10.1016/j.jfoodeng.2013.03.030", "Mercier", "sensitivity analysis drying"),
    ("R4-F21", "Defraeye et al. 2016, Food Bioprod Process（灵敏度与不确定度）",
     "10.1016/j.fbp.2016.08.006", "Defraeye", "uncertainty sensitivity drying"),
    ("R3-F4", "Ni & Datta 1999, Int J Heat Mass Transf（微波对流联合干燥）",
     "10.1016/S0017-9310(98)00229-8", "Ni Datta", "microwave convective drying"),
]


def run(cmd, timeout=60):
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout)
        return r.returncode, r.stdout
    except Exception:
        return 1, b""


def get_json(url, timeout=40):
    code, out = run(["curl", "-sSL", "--ssl-no-revoke", "-A", UA,
                     "--max-time", str(timeout), url], timeout + 20)
    if code != 0 or not out.strip():
        return None
    try:
        return json.loads(out.decode("utf-8", errors="replace"))
    except Exception:
        return None


def openalex_by_doi(doi):
    return get_json(f"https://api.openalex.org/works/doi:{doi}")


def openalex_search(author, phrase):
    q = f"https://api.openalex.org/works?search={phrase.replace(' ', '%20')}"
    q += f"&filter=raw_author_name.search:{author.replace(' ', '%20')}&per-page=5"
    d = get_json(q)
    return (d or {}).get("results") or []


def s2_oa(doi):
    d = get_json("https://api.semanticscholar.org/graph/v1/paper/DOI:"
                 f"{doi}?fields=title,year,openAccessPdf")
    return ((d or {}).get("openAccessPdf") or {}).get("url") if d else None


def oa_urls(work):
    """收集该 work 的所有 OA 候选直链（PDF 优先，其次落地页）。"""
    urls = []
    for loc in (work.get("locations") or []):
        if not loc.get("is_oa"):
            continue
        if loc.get("pdf_url"):
            urls.append(loc["pdf_url"])
        if loc.get("landing_page_url"):
            urls.append(loc["landing_page_url"])
    best = work.get("best_oa_location") or {}
    for u in (best.get("pdf_url"), best.get("landing_page_url")):
        if u:
            urls.insert(0, u)
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def try_download(urls, dest, min_kb=25):
    """逐个候选下载，返回 (成功?, 命中URL, 大小KB)。"""
    for u in urls:
        if not u:
            continue
        if run(["curl", "-sSL", "--ssl-no-revoke", "-A", UA, "--max-time", "90",
                "-o", str(dest), u], 120)[0] == 0 and dest.exists():
            head = dest.read_bytes()[:5]
            kb = dest.stat().st_size // 1024
            if head.startswith(b"%PDF") and kb >= min_kb:
                return True, u, kb
            dest.unlink(missing_ok=True)
    return False, None, 0


def rec_of(key, claimed, doi, work):
    if not work:
        return {"登记键": key, "研报声称": claimed, "doi": doi,
                "核实": "❌ DOI 在 OpenAlex 查不到"}
    src = ((work.get("primary_location") or {}).get("source") or {}).get("display_name")
    return {
        "登记键": key, "研报声称": claimed, "doi": doi, "核实": "✅ DOI 有效",
        "标题": work.get("title"), "年份": work.get("publication_year"),
        "期刊": src,
        "作者": [a["author"]["display_name"]
                 for a in (work.get("authorships") or [])[:4]],
        "是否OA": (work.get("open_access") or {}).get("is_oa"),
        "OA状态": (work.get("open_access") or {}).get("oa_status"),
    }


def match_claimed(work, claimed):
    """DOI 有效但内容对不对？用标题关键词粗判。"""
    t = (work.get("title") or "").lower()
    kw = [w for w in claimed.lower().replace(",", " ").split() if len(w) > 4]
    hit = sum(1 for w in kw if w in t)
    return hit >= 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    PDF.mkdir(parents=True, exist_ok=True)

    man = json.loads((BASE / "引用核实清单.json").read_text(encoding="utf-8"))
    out = []

    # ---------- 一、复检原 DOI，补齐 OA 链接 ----------
    print("=" * 92)
    print("  一、复检 21 条 DOI（双源找 OA）")
    print("=" * 92)
    for r in man:
        doi = r["doi"]
        w = openalex_by_doi(doi)
        if not w:
            r["复核"] = "❌ DOI 无效"
            print(f"\n[{r['登记键']}] {doi}\n   ❌ 查不到")
            out.append(r)
            continue
        claimed = r["研报声称"]
        ok = match_claimed(w, claimed)
        r["标题核对"] = "✅ 与研报声称相符" if ok else "🔴 **与研报声称不符**"
        if not ok:
            r["实际标题"] = w.get("title")
        urls = oa_urls(w)
        s2 = s2_oa(doi)
        if s2:
            urls.insert(0, s2)
        r["OA候选数"] = len(urls)
        print(f"\n[{r['登记键']}] {doi}")
        print(f"   实际: {w.get('title','')[:76]}")
        print(f"   {r['标题核对']}   OA={w.get('open_access',{}).get('is_oa')}  "
              f"候选链接 {len(urls)} 条")
        if not args.check and urls:
            safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in
                           f"{r['登记键']}_{w.get('publication_year')}_"
                           f"{(r.get('作者') or ['anon'])[0]}")[:78]
            dest = PDF / f"{safe.strip()}.pdf"
            got, hit, kb = try_download(urls, dest)
            if got:
                r["已下载"] = dest.name
                r["大小KB"] = kb
                print(f"   ⬇ {dest.name}  {kb} KB")
            else:
                print("   🔒 无可下载的开放全文")
        elif not args.check:
            print("   🔒 无 OA 链接")
        out.append(r)
        time.sleep(0.4)

    # ---------- 二、反查 6 条挂错 DOI 的真身 ----------
    print("\n" + "=" * 92)
    print("  二、反查研报挂错 DOI 的 6 条，找出真正的论文")
    print("=" * 92)
    fixes = []
    for key, claimed, wrong_doi, author, phrase in MISMATCH:
        print(f"\n[{key}] 研报声称: {claimed}")
        print(f"   研报挂的 DOI: {wrong_doi}  ← 指向别的论文")
        res = openalex_search(author, phrase)
        if not res:
            print("   ❌ 未搜到")
            fixes.append({"登记键": key, "研报声称": claimed,
                          "错DOI": wrong_doi, "反查": "未搜到"})
            continue
        for w in res[:3]:
            doi = (w.get("doi") or "").replace("https://doi.org/", "")
            src = ((w.get("primary_location") or {}).get("source")
                   or {}).get("display_name")
            au = [a["author"]["display_name"]
                  for a in (w.get("authorships") or [])[:3]]
            print(f"   · {w.get('title','')[:72]}")
            print(f"     {au} | {src} | {w.get('publication_year')} | {doi}")
        best = res[0]
        bdoi = (best.get("doi") or "").replace("https://doi.org/", "")
        urls = oa_urls(best)
        s2 = s2_oa(bdoi) if bdoi else None
        if s2:
            urls.insert(0, s2)
        item = {"登记键": key, "研报声称": claimed, "错DOI": wrong_doi,
                "反查DOI": bdoi, "反查标题": best.get("title"),
                "反查年份": best.get("publication_year"),
                "反查期刊": ((best.get("primary_location") or {}).get("source")
                          or {}).get("display_name"),
                "反查作者": [a["author"]["display_name"]
                          for a in (best.get("authorships") or [])[:4]],
                "是否OA": (best.get("open_access") or {}).get("is_oa")}
        if not args.check and urls:
            safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in
                           f"{key}_更正_{item['反查年份']}_"
                           f"{(item.get('反查作者') or ['anon'])[0]}")[:78]
            dest = PDF / f"{safe.strip()}.pdf"
            got, hit, kb = try_download(urls, dest)
            if got:
                item["已下载"] = dest.name
                item["大小KB"] = kb
                print(f"   ⬇ {dest.name}  {kb} KB")
            else:
                print("   🔒 无反查到的开放全文")
        fixes.append(item)
        time.sleep(0.5)

    (BASE / "引用核实清单_完整.json").write_text(
        json.dumps({"原DOI复检": out, "挂错DOI反查": fixes},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    n_mis = sum(1 for r in out if "不符" in (r.get("标题核对") or ""))
    n_dl = sum(1 for r in out + fixes if r.get("已下载"))
    print("\n" + "=" * 92)
    print(f"  DOI 有效 {sum(1 for r in out if r.get('核实')=='✅ DOI 有效')}/{len(out)}"
          f" · 其中**与研报声称不符 {n_mis} 条**")
    print(f"  已下载 PDF: {n_dl} 份  → {PDF}")
    print("=" * 92)


if __name__ == "__main__":
    main()
