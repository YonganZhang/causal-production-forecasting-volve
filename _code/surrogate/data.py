"""数据装载与体检。唯一的数据入口——所有模型都从这里拿数据, 保证口径一致。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from contract import (Ledger, check_row_alignment, check_splits, inspect,
                      spotcheck_fields_vs_source, spotcheck_obs_vs_source)

RUNS_ROOT = Path("/mnt/data/yongan-admin-2/datasets/petro/_runs")
PRODUCERS = ("B-1BH", "B-1H", "B-2H", "B-3H", "B-4BH", "B-4DH", "B-4H", "D-1CH", "D-1H",
             "D-2H", "D-3AH", "D-3BH", "D-4AH", "D-4H", "E-1H", "E-2AH", "E-2H", "E-3AH",
             "E-3CH", "E-3H", "E-4AH", "K-3H")
OBS_KEYS = ("WOPR", "WWPR", "WBHP")
FIELD_NAMES = ("PRESSURE", "SWAT")
N_TIMES, N_SNAP = 40, 8
DAY_GRID = np.linspace(1.0, 3312.0, N_TIMES)


def load(dataset: str = "norne", ledger: Ledger | None = None, spotcheck: int = 4):
    """装载 (theta, obs, fields)。ledger 非空时做完整体检 + 回原始文件抽查。"""
    root = RUNS_ROOT / dataset
    pk = root / "packed"
    TH = np.load(pk / "theta.npy").astype(np.float32)
    FL = np.load(pk / "fields.npy")
    OB = np.load(root / "obs_all.npy").astype(np.float32)   # 重抽后的干净标签
    ids = __import__("json").loads((pk / "meta.json").read_text())["case_ids"]
    OB = OB[[int(c[1:]) for c in ids]]                       # 与 packed 的行序对齐

    if ledger is None:
        return TH, OB, FL, ids

    print(f"\n=== 数据体检: {dataset} ===")
    for a, n in ((TH, "theta"), (OB, "obs"), (FL, "fields")):
        inspect(a, n, ledger)
    check_row_alignment(TH, OB, FL, ledger)

    if spotcheck:
        rng = np.random.default_rng(0)
        pick = sorted(rng.choice(len(ids), size=min(spotcheck, len(ids)), replace=False).tolist())
        dirs = [root / ids[i] for i in pick]
        print(f"  回原始文件抽查 {len(pick)} 个样本(不做内部自洽检查):")
        spotcheck_obs_vs_source(pick, dirs, OB, PRODUCERS, OBS_KEYS, DAY_GRID, ledger)
        spotcheck_fields_vs_source(pick, dirs, FL, FIELD_NAMES, N_SNAP, ledger)
    return TH, OB, FL, ids


def splits(TH: np.ndarray, ood_frac: float = 0.20, val_frac: float = 0.15,
           seed: int = 0, ledger: Ledger | None = None) -> dict:
    """与 norne_surrogate.splits 同口径:OOD = |θ| 最大的一批, 其余随机分。"""
    mag = np.linalg.norm(TH, axis=1)
    ood = np.argsort(-mag)[: int(len(TH) * ood_frac)]
    rest = np.setdiff1d(np.arange(len(TH)), ood)
    rng = np.random.default_rng(seed); rng.shuffle(rest)
    n_val = int(len(rest) * val_frac)
    sp = {"train": rest[n_val:], "val": rest[:n_val], "ood": ood}
    if ledger is not None:
        check_splits(TH, sp, ledger)
    return sp
