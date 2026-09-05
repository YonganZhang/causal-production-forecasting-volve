"""代理模型流水线的防复发测试。每一条都锁一个 2026-08-08 审计发现的漏洞。

直接跑:python _tests/test_surrogate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "_code" / "surrogate"))

import metrics as M  # noqa: E402
import models        # noqa: E402


def _toy(seed=0, n=20, c=500):
    rng = np.random.default_rng(seed)
    true = rng.normal(100, 20, (n, 8, 2, c)).astype(np.float32)
    clim = np.repeat(np.median(true, 0)[None], n, axis=0)
    return true, clim


def test_F1_nan_dodging_cannot_win():
    """F1 致命:旧版分子分母掩码不同, 把 95% 输出写成 NaN 能拿 skill=+1.0000。"""
    true, clim = _toy()
    atk = clim.copy()
    rng = np.random.default_rng(1)
    atk[rng.random(atk.shape) < 0.95] = np.nan
    s = M.skill(atk, true, clim)
    assert abs(s) < 1e-6, f"NaN 回避拿到 skill={s:.4f}, 分子分母掩码又不一致了"
    assert M.coverage_frac(atk, true) < 0.1, "覆盖率必须如实反映被掩掉的比例"


def test_climatology_skill_is_exactly_zero():
    """尺子自检:气候态对自己的技巧分在数学上必须恰为 0。"""
    true, clim = _toy()
    assert abs(M.skill(clim, true, clim)) < 1e-12


def test_F5_climatology_uses_median_not_mean():
    """F5:MAE 的最优常数是中位数。用均值会让零输入依赖的预测器拿到正技巧分。"""
    rng = np.random.default_rng(2)
    # 右偏分布下均值与中位数差别明显
    tr = rng.lognormal(3, 1, (60, 8, 2, 200)).astype(np.float32)
    te = rng.lognormal(3, 1, (20, 8, 2, 200)).astype(np.float32)
    o_tr = rng.lognormal(3, 1, (60, 10)).astype(np.float32)
    o_te = rng.lognormal(3, 1, (20, 10)).astype(np.float32)
    cf_med, _ = M.climatology(tr, o_tr, len(te), stat="median")
    cf_mean, _ = M.climatology(tr, o_tr, len(te), stat="mean")
    e_med = np.abs(cf_med - te).mean()
    e_mean = np.abs(cf_mean - te).mean()
    assert e_med < e_mean, "中位数基线的 MAE 应优于均值基线"
    assert M.climatology(tr, o_tr, 1)[0].shape[0] == 1


def test_skill_monotone_in_noise():
    """技巧分必须随注入噪声单调下降。"""
    true, clim = _toy(seed=3)
    rng = np.random.default_rng(4)
    prev = 2.0
    for sigma in (0.5, 2.0, 8.0, 32.0):
        s = M.skill(true + rng.normal(0, sigma, true.shape), true, clim)
        assert s < prev, f"噪声 {sigma} 时技巧分未下降 ({s:.4f} >= {prev:.4f})"
        prev = s


def test_moved_mask_separates_front_from_static():
    """动区掩码必须能把"只在动区准"和"全场准"区分开。"""
    true, clim = _toy(seed=5)
    true[:, -1, 1] = true[:, 0, 1]                 # 先把所有格块设成静止
    true[:, -1, 1, :50] = true[:, 0, 1, :50] + 1.0  # 只有前 50 个动
    mv = M.moved_mask(true)
    assert mv[:, :50].all() and not mv[:, 50:].any()


def test_F6_registry_autodiscovers():
    """F6:文档说"加文件即可", 注册表必须真的自动发现, 不是硬编码 import。"""
    got = models.available()
    for need in ("ridge_pca", "gp", "gbdt", "mlp_pca", "mlp_direct"):
        assert need in got, f"{need} 未被自动发现; 现有 {got}"


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    bad = 0
    for n, f in tests:
        try:
            f(); print(f"  PASS  {n}")
        except Exception as e:  # noqa: BLE001
            bad += 1; print(f"  FAIL  {n}: {e}")
    print(f"\n{len(tests)-bad}/{len(tests)} 通过")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
