#!/usr/bin/env python3
"""唯一的代理模型流水线。

    装载 → 体检(含回原始文件抽查) → 切分 → 尺子校验 → 训练 → 评估 → 落盘

模型是可插拔插件:`--model <name>`。加模型 = 在 models/ 加一个文件实现 fit/predict/describe。

用法:
    python pipeline.py --model ridge_pca                 # 单个模型
    python pipeline.py --all                             # 全部已注册模型
    python pipeline.py --all --dataset norne_wide        # 大扰动批次
    python pipeline.py --audit-only                      # 只做体检与抽查, 不训练
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data as D                     # noqa: E402
import metrics as M                  # noqa: E402
import models                        # noqa: E402
from contract import Ledger, inspect  # noqa: E402

OUT = Path(__file__).resolve().parent.parent.parent / "_pipelines" / "surrogate"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dataset", default="norne")
    ap.add_argument("--spotcheck", type=int, default=4)
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    led = Ledger()
    TH, OB, FL, ids = D.load(args.dataset, ledger=led, spotcheck=args.spotcheck)
    sp = D.splits(TH, ledger=led)

    # 尺子校验:气候态对自己的技巧分必须恰为 0
    print("\n=== 尺子校验(气候态自评, 全部应为 0.0000) ===")
    for tag in ("val", "ood"):
        r = M.selfcheck_ruler(FL[sp["train"]], OB[sp["train"]], FL[sp[tag]], OB[sp[tag]])
        bad = {k: v for k, v in r.items() if np.isfinite(v) and abs(v) > 1e-9}
        print(f"  {tag}: " + "  ".join(f"{k}={v:+.2e}" for k, v in r.items()))
        led.require("ruler", f"climatology_skill_zero/{tag}", not bad, f"非零项 {bad}")

    if args.audit_only:
        print("\n" + led.report())
        led.dump(OUT / f"audit_{args.dataset}{args.tag}.json")
        return 0 if not led.failed else 1

    names = models.available() if args.all else [args.model]
    rows = []
    for nm in names:
        print(f"\n=== 模型 {nm} ===")
        mdl = models.get(nm)()
        t0 = time.time()
        mdl.fit(TH[sp["train"]], FL[sp["train"]], OB[sp["train"]])
        t_fit = time.time() - t0
        row = {"model": nm, "fit_seconds": round(t_fit, 2), "describe": mdl.describe()}
        for tag in ("val", "ood"):
            te = sp[tag]
            t1 = time.time()
            pf, po = mdl.predict(TH[te])
            t_pred = (time.time() - t1) / max(len(te), 1)
            # 🔴 val 与 OOD 都要体检 —— 旧版只查 val, OOD 预测完全没有任何检查
            inspect(pf, f"{nm}/{tag}:pred_fields", led, "predict")
            inspect(po, f"{nm}/{tag}:pred_obs", led, "predict")
            led.require("predict", f"{nm}/{tag}:shape_matches",
                        pf.shape == FL[te].shape and po.shape == OB[te].shape,
                        f"{pf.shape} vs {FL[te].shape}")
            # 预测覆盖率必须是 1:模型不许靠输出 NaN 回避难点(审计实测可骗到 skill=+1.0)
            cov = min(float(np.isfinite(pf).mean()), float(np.isfinite(po).mean()))
            led.require("predict", f"{nm}/{tag}:no_nan_dodging", cov > 0.999,
                        f"预测有限值占比仅 {cov:.4f}, 疑似用 NaN 回避难点")
            cf, co = M.climatology(FL[sp["train"]], OB[sp["train"]], len(te))
            r = M.evaluate(pf, FL[te], po, OB[te], cf, co)
            row[tag] = r
            row[f"{tag}_seconds_per_sample"] = round(t_pred, 6)
        rows.append(row)
        print(f"  val  skill_P={row['val']['skill_P']:+.4f}  skill_S={row['val']['skill_S']:+.4f}  "
              f"skill_obs={row['val']['skill_obs']:+.4f}")
        print(f"  OOD  skill_P={row['ood']['skill_P']:+.4f}  skill_S={row['ood']['skill_S']:+.4f}  "
              f"skill_obs={row['ood']['skill_obs']:+.4f}   单样本前向 {row['ood_seconds_per_sample']*1000:.2f} ms "
              f"(相对 251 秒全模拟加速 {251/max(row['ood_seconds_per_sample'],1e-9):,.0f}×)")

    print(f"\n⚠️  skill 升高≠模型变好:它的分母(气候态误差)也会随分布变化。下表并列绝对 MAE。")
    print(f"{'模型':14s} {'val skP':>9s} {'OOD skP':>9s} {'OOD MAE_P':>10s} {'气候态MAE':>10s} "
          f"{'OOD skS':>9s} {'OOD skObs':>10s} {'ms/样本':>9s}")
    base = next((r for r in rows if r["model"] == "ridge_pca"), None)
    for r in sorted(rows, key=lambda x: -x["ood"]["skill_P"]):
        mark = ""
        if base and r["model"] != "ridge_pca":
            mark = " ✅打过基线" if r["ood"]["skill_P"] > base["ood"]["skill_P"] else " ❌输给基线"
        print(f"{r['model']:14s} {r['val']['skill_P']:>9.4f} {r['ood']['skill_P']:>9.4f} "
              f"{r['ood']['mae_P']:>10.3f} {r['ood']['mae_clim_P']:>10.3f} "
              f"{r['ood']['skill_S']:>9.4f} {r['ood']['skill_obs']:>10.4f} "
              f"{r['ood_seconds_per_sample']*1000:>9.2f}{mark}")

    print("\n" + led.report())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"results_{args.dataset}{args.tag}.json").write_text(
        json.dumps({"dataset": args.dataset, "n": len(TH),
                    "verdict": "ok" if not led.failed else "FAILED",
                    "n_checks": len(led.checks), "n_failed": len(led.failed),
                    "splits": {k: len(v) for k, v in sp.items()},
                    "results": rows}, indent=1, ensure_ascii=False, default=float),
        encoding="utf-8")
    led.dump(OUT / f"audit_{args.dataset}{args.tag}.json")
    return 0 if not led.failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
