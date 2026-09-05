"""MLP 预测 PCA 系数 + 独立 obs 头。

与 ridge_pca 的唯一结构差别: 线性回归 → MLP。切分 / PCA / 评价全部走统一流水线口径。

超参**原样搬自** _sandbox/wf/mlp-pca/:
  prep_pca.py   K=60, svd_solver="randomized", random_state=0, mode="chan"(先按通道量程缩放再 PCA)
  selected.json field: chan / ztgt=global / hidden=128 / depth=2 / dropout=0.1 / wd=1e-2 / lr=1e-3 / mse / epochs=1081
                obs  : perdim / hidden=128 / depth=2 / dropout=0.0 / wd=1e-4 / lr=1e-3 / l1  / epochs=94
  train_mlp.py  _train_full: torch.manual_seed(seed+777), AdamW, CosineAnnealingLR(T_max=epochs),
                bs=min(32,n), 每 epoch 用 CPU generator(seed+777) 重排

seed 固定为 0 —— 原代码 run_search / run_final 的首个 seed(原文最终数字是 seed 0/1/2 三个
指标的**算术平均**; 流水线只能返回一次预测, 因此取原代码的默认 seed, 不做集成、不挑 seed)。

与原代码唯一的口径偏差(已在 describe() 里显式标注):
  通道缩放 sc 原来取**全集** 300 个样本的通道量程(轻微用到了测试集信息)。插件在 fit() 里
  只看得到 train, 故改为 train-only 量程。P 通道量程差 1.9%, 实测对 δ_P 影响 <0.3%。
"""
from __future__ import annotations

import time

import numpy as np

from . import register

K_PCA = 60
FIELD_CFG = dict(hidden=128, depth=2, dropout=0.1, wd=1e-2, lr=1e-3,
                 loss_kind="mse", epochs=1081)
OBS_CFG = dict(hidden=128, depth=2, dropout=0.0, wd=1e-4, lr=1e-3,
               loss_kind="l1", epochs=94)


def _build(torch, nn, d_in, d_out, hidden, depth, dropout):
    layers, d = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.SiLU(), nn.Dropout(dropout)]
        d = hidden
    layers += [nn.Linear(d, d_out)]
    return nn.Sequential(*layers)


def _train_full(X, Y, *, epochs, hidden, depth, dropout, wd, lr, loss_kind, seed, dev):
    """逐行照搬 train_mlp.py::_train_full。"""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed + 777)
    m = _build(torch, nn, X.shape[1], Y.shape[1], hidden, depth, dropout).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    lf = nn.MSELoss() if loss_kind == "mse" else nn.L1Loss()
    n, bs = len(X), min(32, len(X))
    g = torch.Generator(device="cpu").manual_seed(seed + 777)
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(n, generator=g).to(dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            lf(m(X[idx]), Y[idx]).backward()
            opt.step()
        sched.step()
    m.eval()
    return m


@register("mlp_pca")
class MLPPCA:
    def __init__(self, n_components: int = K_PCA, seed: int = 0,
                 field_cfg: dict | None = None, obs_cfg: dict | None = None,
                 sc_from: str = "train"):
        self.k = n_components
        self.seed = seed
        self.fc = dict(FIELD_CFG if field_cfg is None else field_cfg)
        self.oc = dict(OBS_CFG if obs_cfg is None else obs_cfg)
        self.sc_from = sc_from          # "train"(默认) | 显式给定的 (2,) 量程数组
        self._sc_override = None if isinstance(sc_from, str) else np.asarray(sc_from, np.float32)

    # ------------------------------------------------------------------ fit
    def fit(self, X, Yf, Yo):
        import torch
        from sklearn.decomposition import PCA
        t0 = time.time()
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._shape = Yf.shape[1:]

        F = Yf.astype(np.float32)
        if self._sc_override is not None:
            self.sc = self._sc_override
        else:                                        # prep_pca.py 的通道缩放, 但只用 train
            self.sc = np.array([F[:, :, 0].max() - F[:, :, 0].min(),
                                F[:, :, 1].max() - F[:, :, 1].min()], dtype=np.float32)
        Fs = (F / self.sc[None, None, :, None]).reshape(len(F), -1)
        del F

        self.pca = PCA(n_components=min(self.k, len(X) - 1),
                       svd_solver="randomized", random_state=0).fit(Fs)
        self.ev = float(np.cumsum(self.pca.explained_variance_ratio_)[-1])
        Z = self.pca.transform(Fs).astype(np.float32)
        del Fs
        self.zm = Z.mean(0)
        self.zs = np.full(Z.shape[1], Z.std(), dtype=np.float32)   # ztgt="global": 单一全局标量
        T = (Z - self.zm) / self.zs

        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Xn = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32, device=self.dev)

        self.om = Yo.mean(0)
        self.os = Yo.std(0) + 1e-6                                 # otgt="perdim"
        To = torch.tensor((Yo - self.om) / self.os, dtype=torch.float32, device=self.dev)

        fkw = {k: self.fc[k] for k in ("hidden", "depth", "dropout", "wd", "lr", "loss_kind")}
        okw = {k: self.oc[k] for k in ("hidden", "depth", "dropout", "wd", "lr", "loss_kind")}
        self.mf = _train_full(Xn, torch.tensor(T, dtype=torch.float32, device=self.dev),
                              epochs=self.fc["epochs"], seed=self.seed, dev=self.dev, **fkw)
        self.mo = _train_full(Xn, To, epochs=self.oc["epochs"], seed=self.seed,
                              dev=self.dev, **okw)

        # 逆变换所需常量常驻 device(与原代码 run_final 的 fwd() 一致)
        t = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=self.dev)
        self._comp = t(self.pca.components_)
        self._pmean = t(self.pca.mean_)
        self._zs_t, self._zm_t = t(self.zs), t(self.zm)
        self._os_t, self._om_t = t(self.os), t(self.om)
        self._sc_t = t(self.sc)
        self.n_params = sum(p.numel() for p in self.mf.parameters()) + \
            sum(p.numel() for p in self.mo.parameters())
        self.t_fit = time.time() - t0

    # -------------------------------------------------------------- predict
    def predict(self, X):
        import torch
        with torch.no_grad():
            x = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32, device=self.dev)
            z = self.mf(x) * self._zs_t + self._zm_t
            f = (z @ self._comp + self._pmean).reshape((len(X),) + tuple(self._shape))
            f = f * self._sc_t[None, None, :, None]
            o = self.mo(x) * self._os_t + self._om_t
            return (f.cpu().numpy().astype(np.float32),
                    o.cpu().numpy().astype(np.float32))

    def describe(self):
        return {"model": "mlp_pca", "n_components": int(self.pca.n_components_),
                "pca_mode": "chan", "pca_explained_var": round(self.ev, 6),
                "sc_channel_range": [float(v) for v in self.sc],
                "sc_from": "train_only(原代码用全集, 见 docstring)",
                "seed": self.seed, "field_cfg": self.fc, "obs_cfg": self.oc,
                "n_params": int(self.n_params), "device": str(self.dev),
                "fit_seconds": round(self.t_fit, 2)}
