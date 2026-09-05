"""MLP 直接回归全场:theta(13) -> 8*2*44431 场 + 2640 井观测。

从 _sandbox/wf/mlp-direct/mlp_direct.py 迁入, 超参原样搬运 = CFGS[5] "c5_lat16_long800":
    hidden=(64,64) latent=16 dropout=0.0 lr=3e-3 wd=1e-1 epochs=800 bs=16
    chunks=4 w_obs=0.05 seed=0
原始 final.json 里 best_epoch=800(即最后一轮), 所以"取 val 最优检查点"与"取末轮权重"
等价 —— 这里只用 train 训练、取末轮权重, val 完全不参与 fit, OOD 更不参与。
"""
from __future__ import annotations

import os
import time

import numpy as np

# 原实验固定用 6 号卡; pipeline 不带环境变量时也保持同一张卡。
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6")

from . import register  # noqa: E402

CFG = dict(name="c5_lat16_long800", hidden=(64, 64), latent=16, dropout=0.0,
           lr=3e-3, wd=1e-1, epochs=800, bs=16, chunks=4, w_obs=0.05, seed=0)


class _Norm:
    """train-only 统计:场按 per-cell 均值中心化 + per-field 全局 range 缩放;
    obs 按 per-dim 标准化。与原实现逐行一致。"""

    def __init__(self, Ftr32: np.ndarray, Otr_raw: np.ndarray):
        self.f_mean = Ftr32.mean(0)                                  # (8,2,C)
        self.f_rng = np.array([max(float(Ftr32[:, :, i].max() - Ftr32[:, :, i].min()), 1e-9)
                               for i in range(Ftr32.shape[2])], np.float32)
        Otr = np.nan_to_num(Otr_raw, nan=0.0)
        self.o_mean = Otr.mean(0)
        self.o_std = np.maximum(Otr.std(0), 1e-6)

    def f_fwd(self, F):
        return (F - self.f_mean[None]) / self.f_rng[None, None, :, None]

    def f_inv(self, Z):
        return Z * self.f_rng[None, None, :, None] + self.f_mean[None]

    def o_fwd(self, O):
        return (np.nan_to_num(O, nan=0.0) - self.o_mean) / self.o_std

    def o_inv(self, Z):
        return Z * self.o_std + self.o_mean


def _build_net(in_dim, hidden, latent, out_dim, obs_dim, dropout):
    import torch.nn as nn

    class MLPDirect(nn.Module):
        def __init__(self):
            super().__init__()
            layers, d = [], in_dim
            for h in hidden:
                layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout)]
                d = h
            layers += [nn.Linear(d, latent), nn.GELU()]
            self.trunk = nn.Sequential(*layers)
            self.head_f = nn.Linear(latent, out_dim)
            self.head_o = nn.Linear(latent, obs_dim)
            nn.init.zeros_(self.head_f.bias)
            nn.init.zeros_(self.head_o.bias)
            nn.init.normal_(self.head_f.weight, std=0.01)
            self.out_dim, self.obs_dim = out_dim, obs_dim

        def latent(self, x):
            return self.trunk(x)

        def forward(self, x):
            z = self.trunk(x)
            return self.head_f(z), self.head_o(z)

    return MLPDirect()


def _chunked_field(model, z, chunks):
    """z:(B,L) -> 分块产出场输出, 避免一次性物化 (B, 710896) 的中间量。"""
    W, b = model.head_f.weight, model.head_f.bias
    n = model.out_dim
    step = (n + chunks - 1) // chunks
    for s in range(0, n, step):
        e = min(s + step, n)
        yield slice(s, e), z @ W[s:e].T + b[s:e]


@register("mlp_direct")
class MLPDirectModel:
    def __init__(self, **over):
        self.cfg = dict(CFG); self.cfg.update(over)

    # ---------------------------------------------------------------- fit
    def fit(self, X, Yf, Yo):
        import torch

        cfg = self.cfg
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
        t0 = time.time()

        n = len(X)
        self._shape = Yf.shape[1:]                       # (8,2,C)
        out_dim = int(np.prod(self._shape))
        Ftr = Yf.astype(np.float32)
        self.nrm = _Norm(Ftr, Yo)

        # theta 标准化 (train-only)
        self.th_m = X.mean(0)
        self.th_s = np.maximum(X.std(0), 1e-6)
        TH_t = torch.tensor((X - self.th_m) / self.th_s, dtype=torch.float32, device=self.dev)

        Ytr = self.nrm.f_fwd(Ftr).reshape(n, -1)
        Ytr_t = torch.tensor(Ytr, dtype=torch.float16).pin_memory()   # CPU fp16
        Otr_t = torch.tensor(self.nrm.o_fwd(Yo), dtype=torch.float32, device=self.dev)
        del Ytr, Ftr

        self.model = _build_net(X.shape[1], cfg["hidden"], cfg["latent"], out_dim,
                                Yo.shape[1], cfg["dropout"]).to(self.dev)
        self.nparam = sum(p.numel() for p in self.model.parameters())
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

        ch, bs, wo = cfg["chunks"], cfg["bs"], cfg["w_obs"]
        self.hist = []
        for ep in range(cfg["epochs"]):
            self.model.train()
            perm = np.random.permutation(n)
            tot, nb = 0.0, 0
            for i in range(0, n, bs):
                b = perm[i:i + bs]
                x = TH_t[b]
                yb = Ytr_t[b].to(self.dev, non_blocking=True).float()
                ob = Otr_t[b]
                opt.zero_grad(set_to_none=True)
                z = self.model.latent(x)
                loss_o = (self.model.head_o(z) - ob).abs().mean()
                (wo * loss_o).backward(retain_graph=True)
                lf = 0.0
                for sl, o in _chunked_field(self.model, z, ch):
                    w = (sl.stop - sl.start) / out_dim
                    l = (o - yb[:, sl]).abs().mean() * w
                    l.backward(retain_graph=True)
                    lf += float(l.detach())
                del z
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                tot += lf + wo * float(loss_o.detach()); nb += 1
            sched.step()
            if (ep + 1) % 50 == 0:
                self.hist.append((ep + 1, tot / nb))
                print(f"    ep{ep+1:4d} train_L1={tot/nb:.5f}", flush=True)
                torch.cuda.empty_cache() if self.dev == "cuda" else None
        self.t_fit = time.time() - t0

    # ------------------------------------------------------------ predict
    def predict(self, X):
        import torch

        self.model.eval()
        n, ch, bs = len(X), self.cfg["chunks"], 8
        TH_t = torch.tensor((X - self.th_m) / self.th_s, dtype=torch.float32, device=self.dev)
        Fp = np.empty((n,) + self._shape, np.float32)
        Op = np.empty((n, self.model.obs_dim), np.float32)
        with torch.no_grad():
            for i in range(0, n, bs):
                x = TH_t[i:i + bs]
                z = self.model.latent(x)
                Op[i:i + bs] = self.model.head_o(z).float().cpu().numpy()
                buf = np.empty((len(x), self.model.out_dim), np.float32)
                for sl, o in _chunked_field(self.model, z, ch):
                    buf[:, sl] = o.float().cpu().numpy()
                    del o
                Fp[i:i + bs] = buf.reshape((len(x),) + self._shape)
                del z, buf
        return (self.nrm.f_inv(Fp).astype(np.float32),
                self.nrm.o_inv(Op).astype(np.float32))

    def describe(self):
        c = self.cfg
        return {"model": "mlp_direct", "cfg_name": c["name"], "hidden": list(c["hidden"]),
                "latent": c["latent"], "dropout": c["dropout"], "lr": c["lr"], "wd": c["wd"],
                "epochs": c["epochs"], "bs": c["bs"], "chunks": c["chunks"],
                "w_obs": c["w_obs"], "seed": c["seed"], "nparam": int(self.nparam),
                "device": self.dev, "fit_seconds": round(self.t_fit, 2),
                "train_L1_last": round(self.hist[-1][1], 6) if self.hist else None}
