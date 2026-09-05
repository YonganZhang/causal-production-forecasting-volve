"""PCA + 逐分量独立高斯过程 (ARD-RBF)。

架构 (原样搬自 _sandbox/wf/gp/gp_surrogate.py):
    θ(13) --z标准化--> 每个 PCA 分量一个独立 GP --> PCA 系数 --线性反变换--> 场 / 井观测

超参来自原代码 _artifacts/chosen.json (只用 val 选的, OOD 未参与):
    场:  k=80,  各向异性 ARD-RBF, nugget=1e-3
    obs: k=200, 各向同性 RBF,     nugget=1e-3
    PCA 一次拟合到 K_MAX=200 (分量嵌套), 取前 k 个; svd_solver=randomized, random_state=0
    GP:  C(1.0,(1e-3,1e3)) * RBF(ls,(1e-2,1e5)) + White(nugget,(1e-10,1e1))
         normalize_y=True, n_restarts_optimizer=3, random_state=0

这个模型自带不确定度: GP 后验方差经 PCA 线性反变换传播到场/观测空间。
describe() 里额外报告名义 95% 预测区间的实际覆盖率——只用训练集的闭式 LOO
(留一交叉验证) 算, 不碰 val/ood。
"""
from __future__ import annotations

import time
import warnings

import numpy as np

from . import register

SEED = 0
K_MAX = 200          # PCA 分量上限 (train 204 → 最多 203)
K_FIELD = 80         # chosen.json: 场侧 k
K_OBS = 200          # chosen.json: obs 侧 k
ANISO_F = True       # chosen.json: 场侧 ARD (各向异性)
ANISO_O = False      # chosen.json: obs 侧各向同性
NUGGET_F = 1e-3
NUGGET_O = 1e-3
N_RESTARTS = 3
N_JOBS = 48
Z95 = 1.96


def _make_kernel(d: int, aniso: bool, nugget: float):
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    from sklearn.gaussian_process.kernels import ConstantKernel as C
    ls = np.ones(d) if aniso else 1.0
    # 长度尺度上界 1e5: 无关维度可被 ARD 彻底关掉而不撞界
    return (C(1.0, (1e-3, 1e3))
            * RBF(length_scale=ls, length_scale_bounds=(1e-2, 1e5))
            + WhiteKernel(noise_level=nugget, noise_level_bounds=(1e-10, 1e1)))


def _fit_one(Xtr, y, d, aniso, nugget, n_restarts):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.gaussian_process import GaussianProcessRegressor
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    g = GaussianProcessRegressor(kernel=_make_kernel(d, aniso, nugget),
                                 normalize_y=True, n_restarts_optimizer=n_restarts,
                                 random_state=SEED)
    g.fit(Xtr, y)
    return g


def _fit_gps(Xtr, Ztr, aniso, nugget, tag):
    from joblib import Parallel, delayed
    d = Xtr.shape[1]
    t0 = time.time()
    gps = Parallel(n_jobs=min(N_JOBS, Ztr.shape[1]))(
        delayed(_fit_one)(Xtr, Ztr[:, j], d, aniso, nugget, N_RESTARTS)
        for j in range(Ztr.shape[1]))
    print(f"  GP[{tag}] 拟合 {Ztr.shape[1]} 个分量 用时 {time.time()-t0:.1f}s")
    return gps


def _gp_mean(gps, X):
    mu = np.empty((len(X), len(gps)), np.float64)
    for j, g in enumerate(gps):
        mu[:, j] = g.predict(X)
    return mu


def _loo(g):
    """固定超参下 GP 的闭式留一预测 (Rasmussen & Williams 5.12)。

    mu_i = y_i − [K⁻¹y]_i / [K⁻¹]_ii ,  var_i = 1 / [K⁻¹]_ii  (归一化 y 空间)
    """
    from scipy.linalg import cho_solve
    n = len(g.X_train_)
    Kinv = cho_solve((g.L_, True), np.eye(n))
    dg = np.diag(Kinv)
    y = np.asarray(g.y_train_, dtype=np.float64).ravel()
    a = np.asarray(g.alpha_, dtype=np.float64).ravel()
    mu = y - a / dg
    sd = np.sqrt(1.0 / dg)
    ys = float(np.ravel(getattr(g, "_y_train_std", 1.0))[0])
    ym = float(np.ravel(getattr(g, "_y_train_mean", 0.0))[0])
    return mu * ys + ym, sd * ys


def _prop_std(sd_z, comps):
    """PCA 线性反变换的方差传播 (分量后验独立近似): Var(x_j) = Σ_k Var(z_k)·W_kj²。"""
    return np.sqrt(np.maximum(
        (sd_z ** 2).astype(np.float32) @ (comps.astype(np.float32) ** 2), 0.0))


@register("gp")
class GPSurrogate:
    def __init__(self, k_field: int = K_FIELD, k_obs: int = K_OBS):
        self.k_field, self.k_obs = k_field, k_obs

    # ---------------------------------------------------------------- fit
    def fit(self, X, Yf, Yo):
        from sklearn.decomposition import PCA
        t0 = time.time()
        self._shape = Yf.shape[1:]
        n = len(X)

        self.mu_x = X.mean(0)
        self.sd_x = X.std(0) + 1e-12
        Xtr = (X - self.mu_x) / self.sd_x

        F = Yf.astype(np.float32).reshape(n, -1)
        kf_pca = min(K_MAX, n, F.shape[1])
        tp = time.time()
        self.pca_f = PCA(n_components=kf_pca, svd_solver="randomized",
                         random_state=SEED).fit(F)
        print(f"  PCA[field] k={kf_pca} 拟合 {time.time()-tp:.1f}s  "
              f"累计解释方差 = {self.pca_f.explained_variance_ratio_.sum():.6f}")
        Zf = self.pca_f.transform(F)

        O = np.nan_to_num(Yo, nan=0.0)
        ko_pca = min(K_MAX, n, O.shape[1])
        tp = time.time()
        self.pca_o = PCA(n_components=ko_pca, svd_solver="randomized",
                         random_state=SEED).fit(O)
        print(f"  PCA[obs]   k={ko_pca} 拟合 {time.time()-tp:.1f}s  "
              f"累计解释方差 = {self.pca_o.explained_variance_ratio_.sum():.6f}")
        Zo = self.pca_o.transform(O)

        self.kf = min(self.k_field, Zf.shape[1])
        self.ko = min(self.k_obs, Zo.shape[1])
        self.gps_f = _fit_gps(Xtr, Zf[:, :self.kf], ANISO_F, NUGGET_F, "field")
        self.gps_o = _fit_gps(Xtr, Zo[:, :self.ko], ANISO_O, NUGGET_O, "obs")
        self.t_fit = time.time() - t0

        self._cov = self._loo_coverage(F, O)
        del F, O

    # ------------------------------------------------------------ predict
    def predict(self, X):
        Xz = (X - self.mu_x) / self.sd_x
        mz_f = _gp_mean(self.gps_f, Xz)
        mz_o = _gp_mean(self.gps_o, Xz)
        F = self.pca_f.mean_ + mz_f.astype(np.float32) @ self.pca_f.components_[:self.kf]
        O = self.pca_o.mean_ + mz_o.astype(np.float32) @ self.pca_o.components_[:self.ko]
        return (F.reshape((len(X),) + self._shape).astype(np.float32),
                O.astype(np.float32))

    # ----------------------------------------------------------- coverage
    def _loo_coverage(self, F_true, O_true, chunk: int = 24):
        """名义 95% 区间的实际覆盖率, 用训练集闭式 LOO 算 (不碰 val/ood)。

        两种口径:
          gp      —— 只有 GP 后验方差 (漏掉 PCA 截断残差, 必然偏低)
          gp+res  —— 再叠上逐元素截断残差方差 (在同一 LOO 集上标定, 偏乐观)
        """
        t0 = time.time()
        out = {"nominal": 0.95, "method": "closed-form LOO on train split"}

        def _do(gps, pca, k, truth, names, shape_split=None):
            mu_z = np.empty((len(truth), k), np.float64)
            sd_z = np.empty((len(truth), k), np.float64)
            for j, g in enumerate(gps):
                mu_z[:, j], sd_z[:, j] = _loo(g)
            comps = pca.components_[:k]
            # 第一遍: 残差方差 (逐元素)
            res = np.zeros(truth.shape[1], np.float64)
            for s in range(0, len(truth), chunk):
                e = slice(s, min(s + chunk, len(truth)))
                pred = pca.mean_ + mu_z[e].astype(np.float32) @ comps
                res += ((pred - truth[e]).astype(np.float64) ** 2).sum(0)
            res /= len(truth)
            # 第二遍: 覆盖率
            hit = {n: np.zeros(2, np.int64) for n in names}       # [命中, 总数]
            hit_r = {n: np.zeros(2, np.int64) for n in names}
            for s in range(0, len(truth), chunk):
                e = slice(s, min(s + chunk, len(truth)))
                pred = pca.mean_ + mu_z[e].astype(np.float32) @ comps
                sd = _prop_std(sd_z[e], comps)
                sd_r = np.sqrt(sd.astype(np.float64) ** 2 + res[None])
                t = truth[e]
                for nm, sl in names.items():
                    p, tt = pred[:, sl], t[:, sl]
                    for h, s_ in ((hit[nm], sd[:, sl]), (hit_r[nm], sd_r[:, sl])):
                        ok = (tt >= p - Z95 * s_) & (tt <= p + Z95 * s_)
                        h[0] += int(ok.sum()); h[1] += ok.size
            for nm in names:
                out[f"cov95_{nm}_gp"] = round(float(hit[nm][0] / hit[nm][1]), 4)
                out[f"cov95_{nm}_gp+res"] = round(float(hit_r[nm][0] / hit_r[nm][1]), 4)

        # 场: 展平后 (T,2,C) 的通道 0 = P, 通道 1 = S
        T, Ch, C = self._shape
        idx = np.arange(T * Ch * C).reshape(T, Ch, C)
        _do(self.gps_f, self.pca_f, self.kf, F_true,
            {"P": idx[:, 0].ravel(), "S": idx[:, 1].ravel()})
        _do(self.gps_o, self.pca_o, self.ko, O_true,
            {"obs": np.arange(O_true.shape[1])})
        out["seconds"] = round(time.time() - t0, 1)
        print("  覆盖率(训练集闭式 LOO, 名义 95%): "
              + "  ".join(f"{k}={out[k]:.4f}" for k in sorted(out) if k.startswith("cov95_")))
        return out

    # ------------------------------------------------------------ describe
    def describe(self):
        ks_f = self.gps_f[0].kernel_.get_params().get("k1__k2__length_scale")
        return {
            "model": "gp",
            "arch": "PCA + 逐分量独立 GP (ARD-RBF), 后验方差经 PCA 线性传播",
            "k_field": int(self.kf), "k_obs": int(self.ko),
            "k_pca_fit": [int(self.pca_f.n_components_), int(self.pca_o.n_components_)],
            "aniso_field": ANISO_F, "aniso_obs": ANISO_O,
            "nugget_field": NUGGET_F, "nugget_obs": NUGGET_O,
            "n_restarts_optimizer": N_RESTARTS, "normalize_y": True, "seed": SEED,
            "n_gp": int(len(self.gps_f) + len(self.gps_o)),
            "explained_var_field": round(float(self.pca_f.explained_variance_ratio_.sum()), 6),
            "explained_var_obs": round(float(self.pca_o.explained_variance_ratio_.sum()), 6),
            "kernel_field_comp0": str(self.gps_f[0].kernel_),
            "ard_length_scale_comp0": [round(float(v), 4) for v in np.atleast_1d(ks_f)],
            "fit_seconds": round(self.t_fit, 2),
            "uncertainty": self._cov,
            "hparam_source": "_sandbox/wf/gp/_artifacts/chosen.json (val-only 选出)",
        }
