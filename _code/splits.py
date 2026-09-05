"""两种切分口径。

- `well_group_folds`: 留一口井做验证。军伟 T3 用的就是这个, E0 复现必须用它。
  但只有 4~5 口生产井, 每折验证集只有 1 口井, 折间方差极大, 不适合当主口径。
- `rolling_origin_folds`: 同井按时间前推。这才是真实业务场景(用过去预测未来),
  也是本项目 E1 之后的主口径。

滑窗重叠会造成泄漏: step=1 时相邻窗口共享 29 天历史。因此时间切分必须留
embargo(默认 history+horizon 天), 保证训练窗口的任何一天都早于验证窗口的第一天。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from windows import WindowSet


@dataclass
class Fold:
    fold_id: int
    train: np.ndarray   # 布尔掩码
    valid: np.ndarray
    note: str = ""


def well_group_folds(ws: WindowSet, fold_defs: list[dict]) -> list[Fold]:
    """按显式的井分组定义切分, 例如军伟 T3 的 4 折留一井。"""
    folds = []
    for spec in fold_defs:
        train = np.isin(ws.well, list(spec["train_groups"]))
        valid = np.isin(ws.well, list(spec["validation_groups"]))
        folds.append(Fold(int(spec["fold_id"]), train, valid,
                         note=f"val={sorted(spec['validation_groups'])}"))
    return folds


def leave_one_well_out(ws: WindowSet) -> list[Fold]:
    """对窗口集中出现的每口井做一折留一。"""
    wells = sorted(set(ws.well.tolist()))
    return [
        Fold(i, ws.well != w, ws.well == w, note=f"val={w}")
        for i, w in enumerate(wells)
    ]


def rolling_origin_folds(
    ws: WindowSet,
    n_folds: int = 5,
    *,
    embargo_days: int | None = None,
    min_train_frac: float = 0.3,
) -> list[Fold]:
    """时间序 rolling-origin: 按 cutoff 日期把时间轴切成 n_folds 段验证区间。

    第 k 折: 训练 = cutoff <= boundary_k - embargo, 验证 = boundary_k < cutoff <= boundary_{k+1}
    embargo 默认 history+horizon 天, 确保训练窗口不含任何验证窗口用到的日历日。
    """
    if embargo_days is None:
        embargo_days = ws.history_days + ws.horizon
    cutoff = pd.to_datetime(ws.cutoff)
    order = np.sort(cutoff.unique())
    start_idx = int(len(order) * min_train_frac)
    boundaries = np.linspace(start_idx, len(order) - 1, n_folds + 1).astype(int)

    folds = []
    for k in range(n_folds):
        lo, hi = order[boundaries[k]], order[boundaries[k + 1]]
        valid = np.asarray((cutoff > lo) & (cutoff <= hi))
        # 训练窗口的最后一天(cutoff + horizon - 1)必须严格早于验证窗口第一天
        train = np.asarray(cutoff <= (lo - pd.Timedelta(days=embargo_days)))
        if valid.sum() == 0 or train.sum() == 0:
            continue
        folds.append(Fold(k, train, valid,
                          note=f"valid {pd.Timestamp(lo).date()}~{pd.Timestamp(hi).date()}"))
    return folds


def describe(ws: WindowSet, folds: list[Fold]) -> pd.DataFrame:
    rows = []
    for f in folds:
        rows.append({
            "fold": f.fold_id,
            "train_n": int(f.train.sum()),
            "valid_n": int(f.valid.sum()),
            "train_wells": len(set(ws.well[f.train].tolist())),
            "valid_wells": len(set(ws.well[f.valid].tolist())),
            "note": f.note,
        })
    return pd.DataFrame(rows)


def assert_no_calendar_overlap(ws: WindowSet, fold: Fold) -> None:
    """硬检查: 训练窗口覆盖的日历日与验证窗口覆盖的日历日不得相交(同井内)。"""
    cutoff = pd.to_datetime(ws.cutoff)
    for well in set(ws.well[fold.valid].tolist()):
        tr = fold.train & (ws.well == well)
        va = fold.valid & (ws.well == well)
        if not tr.any() or not va.any():
            continue
        train_end = (cutoff[tr] + pd.Timedelta(days=ws.horizon - 1)).max()
        valid_start = (cutoff[va] - pd.Timedelta(days=ws.history_days)).min()
        if train_end >= valid_start:
            raise AssertionError(
                f"fold {fold.fold_id} well {well}: 训练窗口延伸到 {train_end.date()}, "
                f"但验证窗口从 {valid_start.date()} 开始 —— 日历重叠, 存在泄漏"
            )


if __name__ == "__main__":
    from windows import build_windows

    ws = build_windows(step_days=1)
    print("=== rolling-origin (主口径) ===")
    ro = rolling_origin_folds(ws, n_folds=5)
    print(describe(ws, ro).to_string(index=False))
    for f in ro:
        assert_no_calendar_overlap(ws, f)
    print("泄漏检查: 全部通过")
    print("\n=== leave-one-well-out (第二口径) ===")
    print(describe(ws, leave_one_well_out(ws)).to_string(index=False))
