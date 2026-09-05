"""指标。

🔴 主指标是**技巧分**(skill score), 不是归一化 MAE。

原因(实测):δ_P 用全场值域 593 bar 当分母 → 看起来 2.1%; 但真正要预测的信号
(样本间 std)只有 41.6 bar, 误差实际占 29.7%。归一化 MAE 的分母是任选的,
换个分母结论就变, 甚至反号——它不是一个能跨场/井、跨量纲比较的量。

技巧分把分母约掉:
    SS = 1 − err(模型) / err(气候态)
其中气候态 = 直接用训练集均值当预测, 完全无视输入。
    SS = 0   与"什么都不预测"一样好
    SS = 1   完美
    SS < 0   比什么都不预测还差

自检:气候态自己的技巧分**在数学上必须恰好是 0**。算出来不是 0 就说明实现错了。
"""
from __future__ import annotations

import numpy as np


def _mae(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """mask 显式传入时只用它 —— 分子分母必须用**同一个**掩码, 见 skill()。"""
    m = np.isfinite(a) & np.isfinite(b) if mask is None else mask
    return float(np.abs(a[m] - b[m]).mean()) if m.any() else float("nan")


def skill(pred: np.ndarray, true: np.ndarray, clim: np.ndarray) -> float:
    """SS = 1 − MAE(pred) / MAE(clim)。

    🔴 2026-08-08 审计发现的致命漏洞(已修):旧版分子用 isfinite(pred)&isfinite(true)、
    分母用 isfinite(clim)&isfinite(true), 两个掩码不同。于是一个"纯气候态 + 把 95% 输出
    写成 NaN"的假模型, 分子只在最容易的 5% 上算、分母在全部上算, 拿到 **skill=+1.0000
    且体检全绿**。修法:分子分母强制用同一个掩码(三者都有限), 并把被掩掉的比例报出来。
    """
    m = np.isfinite(pred) & np.isfinite(true) & np.isfinite(clim)
    if not m.any():
        return float("nan")
    e_c = _mae(clim, true, m)
    return float("nan") if not np.isfinite(e_c) or e_c <= 0 else 1.0 - _mae(pred, true, m) / e_c


def coverage_frac(pred: np.ndarray, true: np.ndarray) -> float:
    """预测里有多少比例是有限值。低于 1 说明模型在回避难点, 必须与技巧分并列报告。"""
    return float((np.isfinite(pred) & np.isfinite(true)).sum() / max(np.isfinite(true).sum(), 1))


def moved_mask(true_fields: np.ndarray, thresh: float = 0.05) -> np.ndarray:
    """动区:含水饱和度从首帧到末帧变化超过阈值的格块。

    不加这个的话, 大量跨样本根本不动的静止格块会把误差稀释——实测 SWAT 的样本间
    标准差中位数只有 5.6e-5, δ_S=0.6% 绝大部分是稀释出来的。
    """
    t = true_fields.astype(np.float32)
    return np.abs(t[:, -1, 1] - t[:, 0, 1]) > thresh


def evaluate(pred_f: np.ndarray, true_f: np.ndarray,
             pred_o: np.ndarray, true_o: np.ndarray,
             clim_f: np.ndarray, clim_o: np.ndarray) -> dict:
    """返回主指标(技巧分)与次要参考(归一化 MAE, 显式标注分母)。"""
    pf, tf, cf = (x.astype(np.float32) for x in (pred_f, true_f, clim_f))
    out: dict = {}

    for i, nm in enumerate(("P", "S")):
        a, b, c = pf[:, :, i], tf[:, :, i], cf[:, :, i]
        out[f"skill_{nm}"] = skill(a, b, c)
        out[f"mae_{nm}"] = _mae(a, b)
        # 🔴 必须与 skill 并列:skill 升高可能是分母涨得更快, 不代表模型变好。
        # 实测 ridge_pca 从 val 到 OOD, 模型误差涨 23.6% 但气候态涨 30.1%, skill 反而升。
        out[f"mae_clim_{nm}"] = _mae(c, b)
        out[f"pred_coverage_{nm}"] = coverage_frac(a, b)
        # 次要参考:显式记下分母是什么, 免得又被误读
        out[f"delta_{nm}__denom=field_range"] = _mae(a, b) / max(
            float(np.nanmax(b) - np.nanmin(b)), 1e-12)
        out[f"signal_std_{nm}"] = float(b.reshape(len(b), -1).std(axis=0).mean())

    mv = moved_mask(tf)
    if mv.any():
        sel = np.broadcast_to(mv[:, None, :], pf[:, :, 1].shape)
        out["skill_S_front"] = skill(pf[:, :, 1][sel], tf[:, :, 1][sel], cf[:, :, 1][sel])
        out["moved_cell_frac"] = float(mv.mean())

    out["skill_obs"] = skill(pred_o, true_o, clim_o)
    out["mae_obs"] = _mae(pred_o, true_o)
    out["mae_clim_obs"] = _mae(clim_o, true_o)
    out["pred_coverage_obs"] = coverage_frac(pred_o, true_o)
    # 井观测里 61.6% 的条目恰好是 0(关井/未投产, WBHP=0 是哨兵值),
    # 用它们做分母会把相对误差虚增 2.6 倍。所以另报只在"开井"条目上的技巧分。
    act = np.abs(true_o) > 1e-9
    if act.any():
        out["skill_obs_active"] = skill(pred_o[act], true_o[act], clim_o[act])
        out["active_frac"] = float(act.mean())
    return out


def climatology(train_f: np.ndarray, train_o: np.ndarray, n: int,
                stat: str = "median") -> tuple[np.ndarray, np.ndarray]:
    """气候态基线:训练集逐点统计量广播 n 次。

    🔴 stat 默认改为 median:MAE 的最优常数预测是**中位数**不是均值。用均值当分母会让
    一个零输入依赖的预测器拿到 +0.04 而不是 0(审计实测), 即技巧分的零点是偏的。
    """
    f = train_f.astype(np.float32)
    ff = np.median(f, 0) if stat == "median" else f.mean(0)
    oo = np.nanmedian(train_o, 0) if stat == "median" else np.nanmean(train_o, 0)
    return np.repeat(ff[None], n, axis=0), np.repeat(oo[None], n, axis=0)


def selfcheck_ruler(train_f, train_o, test_f, test_o) -> dict:
    """尺子校验:气候态对自己的技巧分必须恰为 0。不是 0 就说明指标实现错了。"""
    cf, co = climatology(train_f, train_o, len(test_f))
    r = evaluate(cf, test_f, co, test_o, cf, co)
    return {k: v for k, v in r.items() if k.startswith("skill")}
