"""
最终下载：对「原 DOI + Crossref 反查到的更正 DOI」逐条找开放全文并下载。"""
import json
import subprocess
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
PDF = BASE / "PDF"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# (标签, DOI, 说明)
TARGETS = [
    ("R3-F3_更正", "10.1016/j.applthermaleng.2006.10.025",
     "Younsi 2007 木材高温热处理计算模型（研报挂错 DOI）"),
    ("R3-F7_更正", "10.1016/s0924-2244(00)00047-9",
     "Champion 2000 玻璃化与分子迁移率（研报挂错 DOI）"),
    ("R4-F18_更正", "10.1016/j.jfoodeng.2011.05.006",
     "Takhar 2011 玉米籽粒水分与应力（研报挂错 DOI）"),
    ("R4-F21_更正Mercier", "10.1016/j.jfoodeng.2013.03.024",
     "Mercier 2013 意面干燥参数灵敏度（研报 DOI 末三位打错）"),
    ("R4-F21_更正Defraeye", "10.1016/j.cherd.2012.06.011",
     "Defraeye 2013 输运性质不确定性对对流干燥的影响（研报挂错 DOI）"),
    ("R3-F4_更正", "10.1205/096030899532475",
     "Ni & Datta 1999（反查到的那篇，主题为油炸非微波）"),
    ("R4-F20_更正", "10.1016/j.jfoodeng.2013.03.030",
     "香蕉圆柱干燥（该 DOI 实为此文，对应 21因素20 而非21）"),
    # 原 DOI 里还没拿到的
    ("R3-F2", "10.1155/2018/9456418", "Vu 2018 多孔介质干燥综述（Hindawi，403）"),
    ("R3-F3", "10.3390/chemengineering4010013", "Häussling Löwgren 2020 Soret（MDPI，403）"),
    ("R3-F5", "10.1016/j.jfoodeng.2015.05.031", "Gulati & Datta 2015 结壳"),
    ("R4-F19", "10.1016/j.jfoodeng.2013.09.014", "Curcio & Aversa 2014 收缩"),
    ("R4-F19", "10.1016/j.jfoodeng.2013.10.015", "Pacheco-Aguirre 2014 各向异性"),
    ("R4-F20", "10.1016/j.jfoodeng.2007.05.013", "Abbasi Souraki 2008 轴向/径向"),
    ("R4-F21", "10.1002/aic.14383", "Mortier 2014 全局灵敏度"),
    ("R3-F7", "10.1016/j.tifs.2005.09.009", "Rahman 2006 食品状态图"),
    ("R2-F8", "10.1080/07373937.2010.545159", "Sami 2011 太阳能柜式干燥机"),
    ("R2-F13", "10.1016/j.applthermaleng.2006.12.020", "Ceylan 2007 热带水果"),
    ("R3-F1", "10.1016/j.jfoodeng.2006.05.013", "Datta 2007 多孔介质 I"),
    ("R3-F7", "10.1016/S0260-8774(99)00039-4", "Bhandari & Howes 1999 玻璃化"),
]


def j(url, timeout=40):
    r = subprocess.run(["curl", "-sSL", "--ssl-no-revoke", "-A", UA,
                        "--max-time", str(timeout), url],
                       capture_output=True, timeout=timeout + 25)
    if r.returncode != 0 or not r.stdout.strip():
        return None
    try:
        return json.loads(r.stdout.decode("utf-8", errors="replace"))
    except Exception:
        return None


def oa_list(doi):
    urls = []
    w = j(f"https://api.openalex.org/works/doi:{doi}")
    if w:
        for loc in (w.get("locations") or []):
            if loc.get("is_oa") and loc.get("pdf_url"):
                urls.append(loc["pdf_url"])
        best = w.get("best_oa_location") or {}
        for u in (best.get("pdf_url"), best.get("landing_page_url")):
            if u:
                urls.insert(0, u)
    s2 = j("https://api.semanticscholar.org/graph/v1/paper/DOI:"
           f"{doi}?fields=openAccessPdf")
    u2 = ((s2 or {}).get("openAccessPdf") or {}).get("url")
    if u2:
        urls.insert(0, u2)
    seen, out = set(), []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return (w.get("title") if w else None), out


def dl(url, dest):
    subprocess.run(["curl", "-sSL", "--ssl-no-revoke", "-A", UA, "--max-time", "90",
                    "-o", str(dest), url], capture_output=True, timeout=120)
    if dest.exists() and dest.read_bytes()[:5] == b"%PDF" and dest.stat().st_size > 25000:
        return dest.stat().st_size // 1024
    dest.unlink(missing_ok=True)
    return 0


def main():
    PDF.mkdir(exist_ok=True)
    ok = []
    for tag, doi, note in TARGETS:
        safe = tag.replace("/", "_")
        dest = PDF / f"{safe}.pdf"
        if dest.exists():
            print(f"  已有 {dest.name}")
            ok.append((tag, doi, dest.name, dest.stat().st_size // 1024, note))
            continue
        title, urls = oa_list(doi)
        print(f"\n[{tag}] {doi}")
        print(f"   {title or '（查不到）'}")
        if not urls:
            print("   🔒 无开放全文")
            continue
        got = 0
        for u in urls:
            got = dl(u, dest)
            if got:
                break
        if got:
            print(f"   ⬇ {dest.name}  {got} KB")
            ok.append((tag, doi, dest.name, got, note))
        else:
            print(f"   🔒 {len(urls)} 条链接均不可下载（多为 403 / 付费墙）")
        time.sleep(0.5)

    print("\n" + "=" * 88)
    print(f"  成功下载 {len(ok)} 份")
    for tag, doi, fn, kb, note in ok:
        print(f"   · {tag:22s} {kb:6d} KB  {fn}")
    print("=" * 88)
    (BASE / "下载记录.json").write_text(
        json.dumps([{"标签": t, "doi": d, "文件": f, "KB": k, "说明": n}
                    for t, d, f, k, n in ok], ensure_ascii=False, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
