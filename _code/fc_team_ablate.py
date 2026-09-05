"""角色消融分析 —— 先测噪声底，再判边际贡献。

🔴 为什么必须先测噪声底:
   第一版按 n=1 得到的消融表看着很漂亮(7 个角色全是正贡献，边际 +10 ~ +148 M$)。
   但把**同一个完整臂**重复跑 5 次，ΔNPV@8% 就在 +158 ~ +249 M$ 之间飘，
   标准差 35.9 M$。也就是说 2σ = 71.8 M$ 以下的"边际贡献"全是 LLM 采样噪声。
   重判之后 7 个角色里只有 2 个可判 —— 其余 5 个的贡献无法与噪声区分。

   这正是「多角色本身不是贡献」的实证形式:复杂度必须由消融买单，买不起就不写。
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fc_water_econ import econ                                   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "_pipelines" / "fc_team"
SIM = ROOT / "_pipelines" / "fc_decide" / "sim"


def best_delta(f: Path, base: dict) -> tuple[float | None, str, dict | None]:
    d = json.loads(f.read_text()); bt = None
    for r in d["rounds"]:
        for m in (r.get("adjudicated") or {}).values():
            if bt is None or m["npv8"] > bt["npv8"]:
                bt = m
    dr = (d.get("dropped") or ["—"])[0]
    return ((bt["npv8"] - base["npv8"]) / 1e6, dr, bt) if bt else (None, dr, None)


def main() -> None:
    base = econ(SIM / "baseline.npz", 2.0, 1.0)
    reps = [P / "loop_A_full.json"] + sorted(P.glob("loop_N*.json"))
    vals = [v for v, _, _ in (best_delta(f, base) for f in reps) if v is not None]
    v = np.array(vals); sd = float(v.std(ddof=1)); thr = 2 * sd
    print(f"噪声底(完整臂 n={len(v)}): 均值 {v.mean():+.1f} sd {sd:.1f} "
          f"范围 {v.min():+.1f}~{v.max():+.1f} M$  → 可分辨阈值 {thr:.1f} M$\n")

    # 🔴 每臂多次重复后用 Welch t 检验，而不是拿单点跟阈值比。
    #    n=1 时 reservoir_engineer 显示 +102.6 M$「显著正贡献」，
    #    补到 n=5 后塌成 +14 M$ 量级 —— 单点消融表不可信。
    arms: dict[str, list[float]] = {}
    for f in sorted(P.glob("loop_[B-H]_*.json")) + sorted(P.glob("loop_R*.json")):
        dv, dr, _ = best_delta(f, base)
        if dv is not None:
            arms.setdefault(dr, []).append(dv)
    rows = []
    for dr, a in arms.items():
        a = np.array(a); n = len(a)
        m, sda = float(a.mean()), float(a.std(ddof=1)) if n > 1 else float("nan")
        mv = float(v.mean() - m)
        se = float(np.sqrt(sd**2 / len(v) + (sda**2 / n if n > 1 else 0.0)))
        t = mv / se if se > 0 else 0.0
        rows.append({"role": dr, "n": n, "mean_MUSD": m, "sd_MUSD": sda,
                     "marginal_MUSD": mv, "t": t,
                     "verdict": ("positive" if t > 2 else
                                 "harmful" if t < -2 else "indistinguishable")})
    print(f"{'剔除角色':<25s}{'n':>3s}{'均值':>10s}{'sd':>7s}{'边际':>9s}{'t':>7s}   判定")
    print("-" * 78)
    for r in sorted(rows, key=lambda x: -x["marginal_MUSD"]):
        tag = {"positive": "✅ 显著正贡献", "harmful": "🔴 显著有害",
               "indistinguishable": "⚪ 噪声内，不可判"}[r["verdict"]]
        print(f"  {r['role']:<25s}{r['n']:>3d}{r['mean_MUSD']:>+9.1f}M{r['sd_MUSD']:>7.1f}"
              f"{r['marginal_MUSD']:>+8.1f}M{r['t']:>+7.2f}   {tag}")

    pooled = float(np.nanmean([r["sd_MUSD"] for r in rows] + [sd]))
    n_need = int(np.ceil(2 * (2 * pooled / 30.0) ** 2))
    out = {"water_price": [2.0, 1.0], "base_npv8": base["npv8"],
           "noise": {"n": len(v), "mean_MUSD": float(v.mean()), "sd_MUSD": sd,
                     "resolvable_2sd_MUSD": thr, "values": vals},
           "ablation": rows,
           "note": (f"n=1/臂。要把可分辨阈值压到 30 M$ 需每臂约 {n_need} 次重复。"
                    "落在噪声内的角色**不得**在论文中写成有贡献。")}
    (P / "ablation.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n合并 sd {pooled:.1f} M$ → 要分辨 30 M$ 级差异每臂约需 {n_need} 次重复")
    print(f"→ {P/'ablation.json'}")


if __name__ == "__main__":
    main()
