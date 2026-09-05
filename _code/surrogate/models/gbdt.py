"""G2 = 岭回归打底 + LightGBM 学残差, 对 PCA 系数逐维回归。

原型: _sandbox/wf/gbdt/{common.py, tune2.py, final.py} 里的 "G2_ridge_plus_gbdt"。
超参原样搬运, 未做任何重新调参:

    场  : perfield PCA k=80, feat=aug, family=ridge_lgbm,
          n_estimators=800 lr=0.02 num_leaves=7 min_child_samples=10
          colsample_bytree=0.6 reg_lambda=0.0 alpha=1.0
    obs : PCA k=80, feat=raw, family=ridge_lgbm,
          n_estimators=200 lr=0.01 num_leaves=31 min_child_samples=20
          colsample_bytree=0.8 reg_lambda=10.0 alpha=0.1

纯 CPU (LightGBM, n_jobs=1)。
"""
from __future__ import annotations

import time

import numpy as np

from . import register

# ---- 原样抄自 _sandbox/wf/gbdt/final.py::MODELS["G2_ridge_plus_gbdt"] ----
CFG = dict(
    field=dict(mode="perfield", k=80, feat="aug", family="ridge_lgbm",
               params=dict(n_estimators=800, learning_rate=0.02, num_leaves=7,
                           min_child_samples=10, colsample_bytree=0.6, reg_lambda=0.0,
                           alpha=1.0)),
    obs=dict(k=80, feat="raw", family="ridge_lgbm",
             params=dict(n_estimators=200, learning_rate=0.01, num_leaves=31,
                         min_child_samples=20, colsample_bytree=0.8, reg_lambda=10.0,
                         alpha=0.1)),
)


def augment(TH):
    """θ 扩展特征: 原始13 + |θ| + 9注入率和 + 4渗透率和 + 注入率×渗透率交叉(36)。

    原样抄自 _sandbox/wf/gbdt/tune2.py::augment。
    """
    r, p = TH[:, :9], TH[:, 9:]
    cross = (r[:, :, None] * p[:, None, :]).reshape(len(TH), -1)
    return np.concatenate([TH, np.linalg.norm(TH, axis=1, keepdims=True),
                           r.sum(1, keepdims=True), p.sum(1, keepdims=True), cross], axis=1)


class FieldCodec:
    """把 (n,8,2,44431) 场编码成 k 维系数, 并能逆变换回去。抄自 common.py。"""

    def __init__(self, mode: str, k: int):
        assert mode in ("joint", "perfield")
        self.mode, self.k = mode, k

    def fit(self, F: np.ndarray):
        from sklearn.decomposition import PCA
        self.shape_ = F.shape[1:]
        X = F.astype(np.float32)
        if self.mode == "joint":
            self.pca_ = PCA(n_components=self.k).fit(X.reshape(len(X), -1))
        else:
            self.mu_, self.sd_, self.pca_l_ = [], [], []
            for i in range(2):                          # PRESSURE, SWAT
                Xi = X[:, :, i].reshape(len(X), -1)
                mu = Xi.mean(); sd = Xi.std() + 1e-9
                self.mu_.append(float(mu)); self.sd_.append(float(sd))
                self.pca_l_.append(PCA(n_components=self.k).fit((Xi - mu) / sd))
                del Xi
        return self

    def encode(self, F: np.ndarray) -> np.ndarray:
        X = F.astype(np.float32)
        if self.mode == "joint":
            return self.pca_.transform(X.reshape(len(X), -1))
        return np.concatenate(
            [self.pca_l_[i].transform((X[:, :, i].reshape(len(X), -1) - self.mu_[i]) / self.sd_[i])
             for i in range(2)], axis=1)

    def decode(self, Z: np.ndarray) -> np.ndarray:
        n = len(Z)
        if self.mode == "joint":
            return self.pca_.inverse_transform(Z).reshape((n,) + self.shape_).astype(np.float32)
        out = np.empty((n,) + self.shape_, dtype=np.float32)
        for i in range(2):
            Zi = Z[:, i * self.k:(i + 1) * self.k]
            rec = self.pca_l_[i].inverse_transform(Zi) * self.sd_[i] + self.mu_[i]
            out[:, :, i] = rec.reshape((n, self.shape_[0], self.shape_[2]))
        return out

    @property
    def n_coef(self) -> int:
        return self.k if self.mode == "joint" else 2 * self.k


def _lgbm(params, seed=0):
    import lightgbm as lgb
    base = dict(objective="regression", n_estimators=300, learning_rate=0.05,
                num_leaves=7, min_child_samples=10, subsample=1.0,
                colsample_bytree=1.0, reg_lambda=0.0, random_state=seed,
                verbose=-1, n_jobs=1, force_col_wise=True)
    base.update(params)
    return lgb.LGBMRegressor(**base)


class MultiOutReg:
    """对每个输出维独立拟合一个回归器。抄自 common.py。"""

    def __init__(self, family: str, params: dict, seed: int = 0):
        self.family, self.params, self.seed = family, dict(params), seed

    def fit(self, X: np.ndarray, Y: np.ndarray):
        from sklearn.linear_model import Ridge
        self.models_ = []
        self.ridge_ = None
        if self.family in ("ridge", "ridge_lgbm"):
            self.ridge_ = Ridge(alpha=self.params.get("alpha", 1.0)).fit(X, Y)
        if self.family == "ridge":
            return self
        R = Y - self.ridge_.predict(X) if self.family == "ridge_lgbm" else Y
        lp = {k: v for k, v in self.params.items() if k != "alpha"}
        if self.family == "lgbm_linear":
            lp = dict(lp); lp["linear_tree"] = True
            lp.setdefault("linear_lambda", 1.0)
        for j in range(Y.shape[1]):
            m = _lgbm(lp, seed=self.seed).fit(X, R[:, j])
            self.models_.append(m)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.family == "ridge":
            return self.ridge_.predict(X)
        P = np.stack([m.predict(X) for m in self.models_], axis=1)
        if self.family == "ridge_lgbm":
            P = P + self.ridge_.predict(X)
        return P


class ObsCodec:
    """井观测的标准化 + PCA 编解码。抄自 common.py。"""

    def __init__(self, k: int):
        self.k = k

    def fit(self, O: np.ndarray):
        from sklearn.decomposition import PCA
        X = np.nan_to_num(O, nan=0.0).astype(np.float32)
        self.mu_ = X.mean(0)
        self.sd_ = X.std(0) + 1e-6
        self.pca_ = PCA(n_components=self.k).fit((X - self.mu_) / self.sd_)
        return self

    def encode(self, O):
        X = np.nan_to_num(O, nan=0.0).astype(np.float32)
        return self.pca_.transform((X - self.mu_) / self.sd_)

    def decode(self, Z):
        return self.pca_.inverse_transform(Z) * self.sd_ + self.mu_

    @property
    def n_coef(self):
        return self.k


@register("gbdt")
class GBDT:
    """G2_ridge_plus_gbdt: 岭回归打底, LightGBM 逐 PCA 系数维学残差。"""

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or CFG

    @staticmethod
    def _feat(TH, mode):
        return TH if mode == "raw" else augment(TH)

    def fit(self, X, Yf, Yo):
        t0 = time.time()
        fc, oc = self.cfg["field"], self.cfg["obs"]
        self.fcod = FieldCodec(fc["mode"], fc["k"]).fit(Yf)
        self.freg = MultiOutReg(fc["family"], fc["params"]).fit(
            self._feat(X, fc["feat"]), self.fcod.encode(Yf))
        self.ocod = ObsCodec(oc["k"]).fit(Yo)
        self.oreg = MultiOutReg(oc["family"], oc["params"]).fit(
            self._feat(X, oc["feat"]), self.ocod.encode(Yo))
        self.t_fit = time.time() - t0
        self.n_sub = len(getattr(self.freg, "models_", [])) + len(getattr(self.oreg, "models_", []))

    def predict(self, X):
        fc, oc = self.cfg["field"], self.cfg["obs"]
        F = self.fcod.decode(self.freg.predict(self._feat(X, fc["feat"])))
        O = self.ocod.decode(self.oreg.predict(self._feat(X, oc["feat"])))
        return F.astype(np.float32), O.astype(np.float32)

    def describe(self):
        return {"model": "gbdt", "variant": "G2_ridge_plus_gbdt",
                "field": self.cfg["field"], "obs": self.cfg["obs"],
                "n_sub_regressors": int(self.n_sub),
                "fit_seconds": round(self.t_fit, 2)}
