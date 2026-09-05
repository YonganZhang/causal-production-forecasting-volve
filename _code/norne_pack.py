#!/usr/bin/env python3
"""把 Norne 批量模拟结果打包成训练张量。

产出（`/mnt/data/yongan-admin-2/datasets/petro/_runs/norne/packed/`）：
    theta.npy    (N, 13)                 输入：9 注入率乘子 + 4 渗透率分区乘子
    obs.npy      (N, 2640)               井观测：22 产井 × {WOPR,WWPR,WBHP} × 40 时刻
    fields.npy   (N, T, 2, 44431) f16    4D 场：T 个快照 × {PRESSURE, SWAT}
    meta.json                            维度、时刻、NaN 统计、失败 case 清单

🔴 纪律（照 petro-knowledge/references/08-陷阱清单.md）：
  - 失败/未收敛的 case **必须登记**，不许静默丢弃 —— 选择性丢弃发散样本会让代理
    只见过"温和"的输入，是最隐蔽的数据泄漏。
  - 缺失值保持 NaN，不填 0。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import os
RUNS = Path(os.environ.get("NORNE_RUNS", "/mnt/data/yongan-admin-2/datasets/petro/_runs/norne"))
OUT = RUNS / "packed"
N_SNAP = 8                       # 从 65 个快照里等间隔取 8 个，控制体积
FIELDS = ("PRESSURE", "SWAT")


def read_fields(unrst: Path, n_snap: int = N_SNAP) -> np.ndarray | None:
    from resdata.resfile import ResdataFile

    if not unrst.exists():
        return None
    f = ResdataFile(str(unrst))
    n = f.num_named_kw(FIELDS[0])
    if n < n_snap:
        return None
    picks = np.linspace(0, n - 1, n_snap).astype(int)
    out = np.stack([np.stack([np.asarray(f[k][i], dtype=np.float32) for k in FIELDS])
                    for i in picks])            # (n_snap, 2, ncell)
    return out.astype(np.float16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=10000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    cases = sorted(RUNS.glob("s[0-9]*"))[: args.max]
    theta, obs, fields, ok_ids, failed = [], [], [], [], []
    obs_all = np.load(RUNS / "obs_all.npy") if (RUNS / "obs_all.npy").exists() else None

    for c in cases:
        j = int(c.name[1:])
        th_p = c / "theta.npy"
        fld = read_fields(c / "out" / "NORNE_ATW2013.UNRST")
        o = obs_all[j] if obs_all is not None and j < len(obs_all) else None
        if not th_p.exists() or fld is None or o is None or not np.isfinite(o).any():
            failed.append({"case": c.name, "reason":
                           "no_theta" if not th_p.exists() else
                           "no_fields" if fld is None else "no_obs"})
            continue
        theta.append(np.load(th_p)); obs.append(o); fields.append(fld); ok_ids.append(c.name)

    if not theta:
        print("没有可用样本"); return 1
    TH = np.stack(theta); OB = np.stack(obs); FL = np.stack(fields)
    np.save(OUT / "theta.npy", TH); np.save(OUT / "obs.npy", OB); np.save(OUT / "fields.npy", FL)

    meta = {
        "n_ok": len(ok_ids), "n_failed": len(failed), "failed": failed,
        "case_ids": ok_ids,
        "theta_shape": list(TH.shape), "obs_shape": list(OB.shape),
        "fields_shape": list(FL.shape), "field_names": list(FIELDS), "n_snapshots": N_SNAP,
        "obs_nan_frac": float(np.isnan(OB).mean()),
        "fields_nan_frac": float(np.isnan(FL.astype(np.float32)).mean()),
        "theta_range": [float(TH.min()), float(TH.max())],
        "pressure_range": [float(np.nanmin(FL[:, :, 0].astype(np.float32))),
                           float(np.nanmax(FL[:, :, 0].astype(np.float32)))],
        "swat_range": [float(np.nanmin(FL[:, :, 1].astype(np.float32))),
                       float(np.nanmax(FL[:, :, 1].astype(np.float32)))],
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"打包完成: {len(ok_ids)} 成功 / {len(failed)} 失败(已登记, 未丢弃)")
    print(f"  theta   {TH.shape}  {TH.dtype}  值域 [{TH.min():+.3f}, {TH.max():+.3f}]")
    print(f"  obs     {OB.shape}  {OB.dtype}  NaN {np.isnan(OB).mean():.2%}")
    print(f"  fields  {FL.shape}  {FL.dtype}  ≈{FL.nbytes/1e9:.2f} GB")
    print(f"    PRESSURE 值域 {meta['pressure_range'][0]:.1f} ~ {meta['pressure_range'][1]:.1f} bar")
    print(f"    SWAT     值域 {meta['swat_range'][0]:.3f} ~ {meta['swat_range'][1]:.3f}")
    if failed:
        print(f"  失败清单: {[f['case'] for f in failed][:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
