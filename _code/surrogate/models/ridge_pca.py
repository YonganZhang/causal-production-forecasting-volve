"""B2 平凡基线:PCA + 岭回归。10 行 sklearn, 所有模型必须打过它。"""
from __future__ import annotations

import time

import numpy as np

from . import register


@register("ridge_pca")
class RidgePCA:
    def __init__(self, n_components: int = 40, alpha: float = 1.0):
        self.k, self.alpha = n_components, alpha

    def fit(self, X, Yf, Yo):
        from sklearn.decomposition import PCA
        from sklearn.linear_model import Ridge
        t0 = time.time()
        self._shape = Yf.shape[1:]
        F = Yf.astype(np.float32).reshape(len(Yf), -1)
        self.pca = PCA(n_components=min(self.k, len(X) - 1)).fit(F)
        self.rg_f = Ridge(alpha=self.alpha).fit(X, self.pca.transform(F))
        self.rg_o = Ridge(alpha=self.alpha).fit(X, np.nan_to_num(Yo))
        self.t_fit = time.time() - t0

    def predict(self, X):
        f = self.pca.inverse_transform(self.rg_f.predict(X)).reshape((len(X),) + self._shape)
        return f.astype(np.float32), self.rg_o.predict(X).astype(np.float32)

    def describe(self):
        return {"model": "ridge_pca", "n_components": int(self.pca.n_components_),
                "alpha": self.alpha, "fit_seconds": round(self.t_fit, 2)}
