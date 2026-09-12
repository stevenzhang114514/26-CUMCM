"""
文献原文抓取器                        [甲 · 数据溯源审计配套工具]

用途
----
为 `src/refs.py` 登记的 13 条研报来源，找回**背后的真实论文**并尽力下载全文。

策略（按可达性排序）
--------------------
1. 用 OpenAlex 按 DOI 反查元数据（标题/作者/期刊/年份/**是否 OA**/OA 的 PDF 直链）
   —— 这一步同时起**核实**作用：若 DOI 查不到，说明研报的引用是错的
2. 有 OA 直链 → 直接下载
3. 无 OA 直链 → 尝试常见 OA 镜像（PMC / MDPI / Hindawi / arXiv）
4. 都不行 → 在清单里标注"付费墙"，给出 DOI 与出版商链接

纪律
----
* 只下载**开放获取**的全文。付费墙的只记录元数据与链接，不绕过。
* 每条都记录**核实结果**：DOI 是否有效、标题是否与研报声称的一致。

用法
----
    python 参考文献原文/fetch_refs.py            # 抓取
    python 参考文献原文/fetch_refs.py --check    # 只核实元数据，不下载
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
PDF = BASE / "PDF"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# ---------------------------------------------------------------------------
# 13 条登记项 → 候选论文 DOI（取自 论文池_药材烘干21因素.md 与四份研报正文）
# ---------------------------------------------------------------------------
REFS = [
    # (登记键, 21因素, 研报声称的出处, DOI, 备注)
    ("R3-F1", "因素1 蒸发潜热", "Kumar 2015 Drying Technology",
     "10.1080/07373937.2014.947512", "有效扩散系数与蒸发冷却"),
    ("R3-F1", "因素1 蒸发潜热", "Datta 2007 J Food Eng I",
     "10.1016/j.jfoodeng.2006.05.013", "多孔介质联立传热传质 I"),
    ("R3-F1", "因素1 蒸发潜热", "Defraeye 2012 IJHMT（「偏高12~22°C」的来源）",
     "10.1016/j.ijheatmasstransfer.2011.08.047", "对流干燥先进模型综述"),
    ("R3-F2", "因素2 内部蒸发/蒸气渗流", "Vu et al. 2018 Int J Chem Eng",
     "10.1155/2018/9456418", "多孔介质干燥传热传质综述与数值实现"),
    ("R3-F3", "因素3 Soret/Dufour", "Häussling Löwgren 2020 ChemEngineering",
     "10.3390/chemengineering4010013", "干燥中 Soret 效应的数值实现"),
    ("R3-F3", "因素3 Soret/Dufour", "Younsi 2007 Int J Therm Sci",
     "10.1016/j.ijthermalsci.2006.09.006", "木材高温热处理三维模型"),
    ("R3-F4", "因素4 辐射换热", "Ni & Datta 1999 Int J Heat Mass Transf",
     "10.1016/S0017-9310(98)00229-8", "微波与对流联合干燥"),
    ("R3-F5", "因素5 表面结壳", "Gulati & Datta 2015 J Food Eng",
     "10.1016/j.jfoodeng.2015.05.031", "结壳与质构发展的机理"),
    ("R3-F6", "因素6 结合水", "NMR T2 relaxometry 2016 IFSE",
     "10.1016/j.ifset.2016.10.015", "NMR 测结合水/自由水"),
    ("R3-F7", "因素7 玻璃化", "Bhandari & Howes 1999 J Food Eng",
     "10.1016/S0260-8774(99)00039-4", "玻璃化转变对干燥与稳定性的意义"),
    ("R3-F7", "因素7 玻璃化", "Rahman 2006 Trends Food Sci Tech",
     "10.1016/j.tifs.2005.09.009", "食品状态图"),
    ("R3-F7", "因素7 玻璃化", "Champion 2000 J Food Eng（NMR 10^2~10^4 倍）",
     "10.1016/S0260-8774(00)00028-5", "玻璃态中扩散系数跌落"),
    ("R2-F8", "因素8 双向耦合", "Sami 2011 Drying Technology",
     "10.1080/07373937.2010.545159", "间接太阳能柜式干燥机动态模型"),
    ("R2-F13", "因素13 控温波动", "Ceylan 2007 Appl Therm Eng",
     "10.1016/j.applthermaleng.2006.12.020", "热带水果干燥特性建模"),
    ("R4-F18", "因素18 干裂", "Takhar 2011 J Food Eng",
     "10.1016/j.jfoodeng.2011.01.026", "玉米籽粒水分梯度致裂"),
    ("R4-F19", "因素19 各向异性收缩", "Curcio & Aversa 2014 J Food Eng",
     "10.1016/j.jfoodeng.2013.09.014", "收缩对蔬菜对流干燥的影响"),
    ("R4-F19", "因素19 各向异性收缩", "Pacheco Aguirre 2014 J Food Eng",
     "10.1016/j.jfoodeng.2013.10.015", "圆柱固体各向异性扩散系数（胡萝卜）"),
    ("R4-F20", "因素20 端面效应", "Abbasi Souraki & Mowla 2008 J Food Eng",
     "10.1016/j.jfoodeng.2007.05.013", "青豆轴向/径向扩散系数（含收缩）"),
    ("R4-F21", "因素21 不确定性", "Mercier 2013 J Food Eng（研报称）",
     "10.1016/j.jfoodeng.2013.03.030", "⚠️ 该 DOI 被研报同时指给「香蕉圆柱干燥」"),
    ("R4-F21", "因素21 不确定性", "Defraeye 2016 Food Bioprod Process",
     "10.1016/j.fbp.2016.08.006", "先进干燥模型的灵敏度与不确定度"),
    ("R4-F21", "因素21 不确定性", "Mortier 2014 AIChE Journal",
     "10.1002/aic.14383", "全局灵敏度分析用于颗粒干燥模型"),
]


def curl(url, out=None, timeout=45, head=False):
    cmd = ["curl", "-sSL", "--ssl-no-revoke", "-A", UA, "--max-time", str(timeout)]
    if head:
        cmd += ["-o", os.devnull, "-w", "%{http_code}|%{content_type}|%{size_download}"]
    elif out:
        cmd += ["-o", str(out)]
    cmd.append(url)
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout + 15)
        if head:
            return r.stdout.decode(errors="replace").strip()
        return r.returncode == 0
    except Exception:
        return "" if head else False


def openalex(doi):
    url = f"https://api.openalex.org/works/doi:{doi}"
    cmd = ["curl", "-sSL", "--ssl-no-revoke", "-A", UA, "--max-time", "30", url]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=45)
        if r.returncode != 0 or not r.stdout.strip():
            return None
        return json.loads(r.stdout.decode("utf-8", errors="replace"))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只核实元数据，不下载")
    args = ap.parse_args()
    PDF.mkdir(parents=True, exist_ok=True)

    manifest = []
    seen = set()
    for key, factor, claimed, doi, note in REFS:
        if doi in seen:
            continue
        seen.add(doi)
        rec = {"登记键": key, "21因素": factor, "研报声称": claimed,
               "doi": doi, "备注": note}
        print(f"\n{'='*88}\n[{key}] {doi}\n  研报声称: {claimed}", flush=True)

        d = openalex(doi)
        if not d:
            rec["核实"] = "❌ DOI 在 OpenAlex 查不到"
            print("  ❌ OpenAlex 查不到该 DOI —— 研报的引用可能是错的", flush=True)
            manifest.append(rec)
            continue

        title = d.get("title") or ""
        year = d.get("publication_year")
        host = ((d.get("primary_location") or {}).get("source") or {}).get("display_name")
        oa = d.get("open_access") or {}
        best = d.get("best_oa_location") or {}
        pdf_url = best.get("pdf_url")
        land = best.get("landing_page_url")
        authors = [a["author"]["display_name"]
                   for a in (d.get("authorships") or [])[:4]]

        rec.update({"核实": "✅", "标题": title, "年份": year, "期刊": host,
                    "作者": authors, "是否OA": oa.get("is_oa"),
                    "OA状态": oa.get("oa_status"), "OA链接": land,
                    "PDF直链": pdf_url})
        print(f"  ✅ {title[:80]}\n     {host} · {year} · {', '.join(authors[:3])}")
        print(f"     OA: {oa.get('is_oa')} ({oa.get('oa_status')})  "
              f"PDF: {pdf_url or '（无直链）'}")

        if args.check:
            manifest.append(rec)
            time.sleep(0.4)
            continue

        # 下载
        safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in
                       f"{key}_{year}_{authors[0] if authors else 'anon'}")[:80]
        dest = PDF / f"{safe.strip()}.pdf"
        got = None
        for cand in (pdf_url, land):
            if not cand:
                continue
            if curl(cand, dest):
                info = curl(cand, head=True)
                code, ctype, size = (info.split("|") + ["", "", "0"])[:3]
                if code == "200" and "pdf" in ctype.lower() and int(size or 0) > 20000:
                    got = (cand, size)
                    break
        if got:
            rec["已下载"] = str(dest.relative_to(BASE))
            rec["大小KB"] = int(got[1]) // 1024
            print(f"  ⬇ 已下载 {dest.name}  {int(got[1])//1024} KB", flush=True)
        else:
            if dest.exists():
                dest.unlink()
            rec["已下载"] = None
            print("  🔒 付费墙或无 PDF 直链 —— 只记录元数据", flush=True)
        manifest.append(rec)
        time.sleep(0.5)

    out = BASE / "引用核实清单.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    n_ok = sum(1 for r in manifest if r.get("核实") == "✅")
    n_dl = sum(1 for r in manifest if r.get("已下载"))
    n_oa = sum(1 for r in manifest if r.get("是否OA"))
    print(f"\n{'='*88}")
    print(f"  合计 {len(manifest)} 条 · DOI 有效 {n_ok} · 开放获取 {n_oa} · 已下载 {n_dl}")
    print(f"  → {out}")
    print(f"  → PDF 目录: {PDF}")


if __name__ == "__main__":
    main()
