"""留一角色消融的对比与裁定 —— 用来定"最强的 Agent Team 到底是哪几个角色"。

    python _code/gaia_ablate.py

口径与 gaia.py 完全一致(同一基准、同一 ΔNPV 定义、同样剔除被否决方案)。
只比较**同一配置(F2)下**跑出来的批次;跨配置的旧批次不参与裁定。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats as st

import gaia

ROOT = Path(__file__).resolve().parent.parent
#: 消融批次前缀 → 剔除的角色。tag 规则见 run_ablate_queue.sh。
ABL = {
    "G2nogeom": "geomechanics_expert",
    "G2noecon": "economics_analyst",
    "G2noseis": "seismic_4d_analyst",
    "G2nocon": "connectivity_analyst",
    "G2nopro": "production_surveillance",
    "G2nores": "reservoir_engineer",
    "G2nocons": "constraint_auditor",
}


def _runs(prefix: str) -> list[float]:
    """该配置每次实验的最好 ΔNPV(百万美元)。"""
    base = gaia.baseline_npv()
    out = []
    for f in sorted(gaia.LOOP_DIR.glob(f"loop_{prefix}_*.json")):
        d = json.loads(f.read_text())
        v = [(m["npv8"] - base) / 1e6
             for r in d["rounds"]
             for k, m in (r.get("adjudicated") or {}).items()
             if int(k) not in set(r.get("vetoed") or [])]
        if v:
            out.append(max(v))
    return out


def summary() -> dict:
    full = np.array(gaia.arm("full7")["values"])
    rows = []
    for pre, role in ABL.items():
        v = np.array(_runs(pre))
        if len(v) < 2:
            continue
        t = st.ttest_ind(v, full, equal_var=False)
        ci = st.t.interval(0.95, len(v) + len(full) - 2,
                           loc=v.mean() - full.mean(),
                           scale=np.sqrt(v.var(ddof=1) / len(v) + full.var(ddof=1) / len(full)))
        rows.append({
            "dropped": role, "n": len(v), "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)), "best": float(v.max()),
            "diff_vs_full": float(v.mean() - full.mean()),
            "ci": [float(ci[0]), float(ci[1])], "p": float(t.pvalue),
            "sim_budget": _budget(pre),
        })
    rows.sort(key=lambda r: -r["mean"])
    return {"full": {"n": len(full), "mean": float(full.mean()),
                     "sd": float(full.std(ddof=1)), "best": float(full.max()),
                     "sim_budget": gaia.arm("full7")["sim_budget"]},
            "ablations": rows}


def _budget(prefix: str) -> float:
    n = []
    for f in sorted(gaia.LOOP_DIR.glob(f"loop_{prefix}_*.json")):
        d = json.loads(f.read_text())
        n.append(sum(len(r.get("adjudicated") or {}) for r in d["rounds"]))
    return float(np.mean(n)) if n else float("nan")


def report() -> str:
    s = summary()
    f = s["full"]
    L = [f"基准 NPV@8% = {gaia.baseline_npv()/1e6:.1f} M$   ΔNPV 越高越好   "
         f"(与 gaia.py 同口径,只比 F2 配置)",
         "=" * 92,
         f"{'配置':<26s}{'n':>3s}{'均值':>10s}{'sd':>8s}{'最好':>10s}"
         f"{'vs 全角色':>11s}{'95% CI':>20s}{'p':>8s}",
         "-" * 92,
         f"{'全 7 角色 (Gaia)':<26s}{f['n']:>3d}{f['mean']:>+9.1f}M{f['sd']:>8.1f}"
         f"{f['best']:>+9.1f}M{'—':>11s}{'—':>20s}{'—':>8s}"]
    for r in s["ablations"]:
        lo, hi = r["ci"]
        ci_txt = f"[{lo:+.0f}, {hi:+.0f}]"
        name = "去掉 " + r["dropped"]
        L.append(f"{name:<26s}{r['n']:>3d}{r['mean']:>+9.1f}M{r['sd']:>8.1f}"
                 f"{r['best']:>+9.1f}M{r['diff_vs_full']:>+10.1f}M{ci_txt:>20s}"
                 f"{r['p']:>8.3f}")
    L += ["=" * 92, "",
          "裁定规则:只有 95% CI 完全不含 0,才算证据支持「该去掉这个角色」;",
          "否则维持全 7 角色 —— 不能因为均值高一点就删角色(LLM 输出本身有波动)。"]
    return "\n".join(L)


if __name__ == "__main__":
    import sys
    print(json.dumps(summary(), ensure_ascii=False, indent=2) if "--json" in sys.argv else report())
