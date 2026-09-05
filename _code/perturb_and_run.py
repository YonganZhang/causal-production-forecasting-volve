#!/usr/bin/env python3
"""G2 可辨识性诊断:扰动油藏参数,量化每个观测量的响应强度。

问题:历史拟合是从产量数据反推地下参数。但**只有对参数有响应的观测量才携带信息**。
如果某个观测量对渗透率怎么变都不动,拟合它等于拟合噪声。

方法:插入 MULTIPLY 全局缩放 PERMX/PERMY/PERMZ(三个都乘),跑几个乘子,
测量每个井级观测量的响应幅度。

🔴 2026-08-07 审计修正三点:
  1. 旧 docstring 说"不动 PERMZ"但代码乘了 PERMZ —— 现改为文档与实现一致(三个都乘)。
  2. MULTIPLY 插在 MAXVALUE 之前, 于是 PERMX 20000 / PERMZ 2000 的上限会把一部分单元
     截回原值, 标称乘子并未完全施加。make_case 现在会报告被截断的比例。
  3. 旧 docstring 承诺"噪声地板 + 响应/噪声>100 判据"但从未实现, 落盘 JSON 也没有
     pass/fail 字段。现已实现噪声地板估计并写入 JSON。

用法:
    python perturb_and_run.py --mult 0.5 1.0 2.0 --tag permx
    python perturb_and_run.py --analyze          # 只分析已有结果
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parent.parent
BASE = PROJ / "_sandbox" / "opm_run"
OUT = PROJ / "_sandbox" / "g2_runs"
IMAGE = "openporousmedia/opmreleases:latest"
PRODUCERS = ("P-F-12", "P-F-14", "P-F-11B", "P-F-15D", "P-F-1C")
OBSERVABLES = ("WOPR", "WWPR", "WGPR", "WBHP", "WWCT")


def make_case(mult: float, tag: str) -> Path:
    """复制 deck 并插入全局渗透率乘子。插在 MAXVALUE 之前, 保证截断仍生效。"""
    case = OUT / f"{tag}_{mult:g}"
    if case.exists():
        shutil.rmtree(case)
    shutil.copytree(BASE, case, ignore=shutil.ignore_patterns("out", "*.orig"))
    deck = case / "VOLVE_2016.DATA"
    lines = deck.read_text(errors="ignore").split("\n")
    idx = next(i for i, l in enumerate(lines) if l.strip().upper().startswith("MAXVALUE"))
    block = [
        f"-- [G2 扰动] 全局渗透率乘子 {mult:g}",
        "MULTIPLY",
        f"  PERMX  {mult:g}  1 108  1 100  1 63 /",
        f"  PERMY  {mult:g}  1 108  1 100  1 63 /",
        f"  PERMZ  {mult:g}  1 108  1 100  1 63 /",
        "/",
        "",
    ]
    deck.write_text("\n".join(lines[:idx] + block + lines[idx:]))
    (case / "out").mkdir(exist_ok=True)
    report_clipping(mult, case)
    return case


def report_clipping(mult: float, case: Path) -> dict:
    """报告 MAXVALUE 会把多少比例的单元截回上限 —— 标称乘子未完全施加的部分。"""
    from resdata.resfile import ResdataFile

    init = BASE / "out" / "VOLVE_2016.INIT"
    if not init.exists():
        return {}
    f = ResdataFile(str(init))
    info = {}
    for kw, cap in (("PERMX", 20000.0), ("PERMZ", 2000.0)):
        a = np.asarray(f[kw][0], dtype=float)
        info[kw] = float(((a * mult) > cap).mean())
    (case / "clipping.json").write_text(json.dumps(
        {"mult": mult, "fraction_clipped": info}, indent=1), encoding="utf-8")
    return info


def run(case: Path, threads: int = 8) -> int:
    cmd = ["docker", "run", "--rm", "-u", f"{os.getuid()}:{os.getgid()}",
           "-v", f"{case}:/data", "-w", "/data", IMAGE,
           "flow", "VOLVE_2016.DATA", "--output-dir=/data/out",
           "--parsing-strictness=low", f"--threads-per-process={threads}"]
    with open(case / "run.log", "w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=7200).returncode


def read_case(path: Path) -> dict | None:
    from resdata.summary import Summary

    smry = path / "out" / "VOLVE_2016.UNSMRY"
    if not smry.exists():
        return None
    s = Summary(str(smry))
    out = {}
    for k in OBSERVABLES:
        for w in PRODUCERS:
            try:
                out[f"{k}:{w}"] = np.asarray(s.numpy_vector(f"{k}:{w}"), dtype=float)
            except Exception:                     # 该井没有这个量
                pass
    out["_time"] = np.asarray(s.numpy_vector("TIME"), dtype=float)
    return out


# 各 case 的自适应时间步不同(实测 424/433/445 个报告点), 直接按下标相减
# 比的是相差最多 115 天的两个时刻 —— 2026-08-07 审计发现的致命错误。
# 统一插值到同一时间网格后再比。
GRID = np.linspace(1.0, 3197.0, 400)


def _on_grid(t: np.ndarray, v: np.ndarray) -> np.ndarray:
    m = np.isfinite(t) & np.isfinite(v)
    if m.sum() < 10:
        return np.full(len(GRID), np.nan)
    return np.interp(GRID, t[m], v[m], left=np.nan, right=np.nan)


def analyze(tag: str = "permx") -> dict:
    """响应幅度 vs 数值噪声地板。噪声地板用两个最接近的乘子之间的差分近似。"""
    cases = sorted(OUT.glob(f"{tag}_*"), key=lambda p: float(p.name.split("_")[-1]))
    data = {c.name: read_case(c) for c in cases}
    data = {k: v for k, v in data.items() if v}
    base_name = next((k for k in data if k.endswith("_1")), None)
    if base_name is None or len(data) < 2:
        return {"error": f"需要至少 2 个成功 case(含乘子=1), 现有 {list(data)}"}

    base = data[base_name]
    rows = []
    for key in sorted(k for k in base if not k.startswith("_")):
        b = _on_grid(base["_time"], base[key])
        m = np.isfinite(b) & (np.abs(b) > 1e-9)
        if m.sum() < 20:
            continue
        scale = np.abs(b[m]).mean()
        resp = {}
        for name, d in data.items():
            if name == base_name or key not in d:
                continue
            a = _on_grid(d["_time"], d[key])
            mm = m & np.isfinite(a)
            if mm.sum() < 20:
                continue
            resp[name] = float(np.abs(a[mm] - b[mm]).mean() / scale)
        if resp:
            rows.append({"observable": key, "rel_response": resp,
                         "max_response": max(resp.values())})
    rows.sort(key=lambda r: -r["max_response"])
    # 数值噪声地板:用同一 case 的相邻时间点差分的中位数近似(同一物理解, 差异只能来自数值)
    # 噪声地板必须在**原始**报告点上估 —— 插值到均匀网格会把数值抖动抹平(实测 SNR 变 inf)。
    floor = {}
    for key in sorted(k for k in base if not k.startswith("_")):
        raw = base[key]
        m = np.isfinite(raw)
        if m.sum() < 20:
            continue
        scale = np.abs(raw[m]).mean()
        d2 = np.abs(np.diff(raw[m], n=2))        # 二阶差分滤掉趋势, 留下步长抖动
        floor[key] = float(np.median(d2) / max(scale, 1e-12))
    for r in rows:
        f = floor.get(r["observable"], np.nan)
        r["noise_floor"] = f
        r["snr"] = float(r["max_response"] / f) if f and np.isfinite(f) and f > 0 else float("inf")
        r["identifiable"] = bool(r["snr"] > 100)
    n_id = sum(1 for r in rows if r["identifiable"])
    return {"base": base_name, "n_cases": len(data), "rows": rows,
            "criterion": "snr = max_response / noise_floor > 100",
            "n_identifiable": n_id, "n_observables": len(rows)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mult", type=float, nargs="*", default=[])
    ap.add_argument("--tag", default="permx")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    for m in args.mult:
        case = make_case(m, args.tag)
        print(f"[run] {case.name} ...", flush=True)
        rc = run(case, args.threads)
        print(f"[run] {case.name} exit={rc}", flush=True)

    if args.analyze or args.mult:
        res = analyze(args.tag)
        p = PROJ / "_pipelines" / "g0_g2_gates" / f"g2_sensitivity_{args.tag}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        if "rows" in res:
            print(f"\n{'观测量':16s} " + " ".join(f"{k.split('_')[-1]:>10s}" for k in
                                                sorted(res['rows'][0]['rel_response'])))
            for r in res["rows"]:
                cells = " ".join(f"{r['rel_response'][k]:>10.2%}"
                                 for k in sorted(r["rel_response"]))
                print(f"{r['observable']:16s} {cells}  SNR={r['snr']:>8.1f} "
                      f"{'✅' if r['identifiable'] else '❌'}")
            print(f"\n判据 {res['criterion']}: {res['n_identifiable']}/{res['n_observables']} 通过")
        print(f"\n已写入 {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
