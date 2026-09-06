"""代理模型包装 —— 给智能体团队提供**毫秒级**方案评估。

🔴 定位(2026-08-30 用户纠正):代理不是"压缩搜索空间"的启发式，
   它是**油藏数值模拟器的快速替身**：一次评估 0.05 ms vs 模拟约 1 分钟。
   团队因此可以每轮试上千个方案，而模拟器只用来裁定少数候选。

🔴 但代理不能当裁判。实测:随机方案上误差 0.37%，可是在优化器找到的极值点上
   把目标差高估 4.1 倍，代理排第 1 的真跑最差、排第 4 的真跑最好。
   所以本模块只输出**筛选用分数**，最终取值一律以 OPM Flow 的 field_cum 为准。
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
import forecast_gen as FG          # noqa: E402
import norne_bulk as NB            # noqa: E402
from fc_mech import PosNet         # noqa: E402

CKPT = ROOT / "_pipelines" / "fc_rank_fidelity" / "surrogate.pt"
NORM = ROOT / "_pipelines" / "fc_team" / "norm.npz"
N_PIN, N_TRAIN, NT = 8000, 7400, FG.FC_N
DAYS = np.asarray(FG.FC_GRID, float)
WT = np.gradient(DAYS); WT[0] *= .5; WT[-1] *= .5          # 梯形权重
MS3 = (DAYS - DAYS[0]) <= 3 * 365.25


def _build_norm() -> None:
    """复算训练集归一化常量。必须与 fc_rank_fidelity 的切分逐位一致。"""
    sh = sorted((FG.OUT / "shards").glob("*.npz"))[:N_PIN]
    TH, Y = [], []
    for p in sh:
        z = np.load(p); ob = z["obs"]
        TH.append(z["theta"].ravel())
        Y.append(np.stack([[ob[(i*3+k)*NT:(i*3+k+1)*NT] for k in (0, 1)]
                           for i in range(len(NB.PRODUCERS))]))
    TH = np.stack(TH).astype(np.float32); Y = np.stack(Y).astype(np.float32)
    live = ~(np.abs(Y).max(axis=(0, 3)) < 1e-9).all(1)
    Y = Y[:, live]
    idx = np.random.default_rng(0).permutation(len(TH))     # 与训练同 seed
    tr = idx[600:][:N_TRAIN]
    Yf = Y.reshape(len(TH), -1)
    NORM.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(NORM, live=live,
                        xm=TH[tr].mean(0), xs=TH[tr].std(0) + 1e-8,
                        ym=Yf[tr].mean(0), ys=Yf[tr].std(0) + 1e-8)
    print(f"归一化常量已缓存 → {NORM.name}  活井 {int(live.sum())}")


class Proxy:
    def __init__(self, device: str = "cpu"):
        if not NORM.exists():
            _build_norm()
        z = np.load(NORM)
        self.live = z["live"]; self.nw = int(self.live.sum())
        self.xm, self.xs, self.ym, self.ys = z["xm"], z["xs"], z["ym"], z["ys"]
        self.dev = device
        self.net = PosNet(24, self.nw * 2, NT, "fourier", n_bands=16).to(device)
        self.net.load_state_dict(torch.load(CKPT, map_location=device))
        self.net.eval()

    @torch.no_grad()
    def trajectories(self, theta: np.ndarray) -> np.ndarray:
        """theta: (B, 24) 或 (24,) → (B, n_live_well, 2, NT) 逐井产油/产水轨迹。

        反归一化只在这里做一次;`evaluate` 与绘图都走它,避免同一个量两套算法。
        """
        th = np.atleast_2d(np.asarray(theta, np.float32))
        x = torch.tensor((th - self.xm) / self.xs, device=self.dev)
        y = self.net(x).cpu().numpy().reshape(len(th), -1)
        y = y * self.ys + self.ym                      # 反归一化
        return y.reshape(len(th), self.nw, 2, NT)

    def evaluate(self, theta: np.ndarray) -> dict:
        """theta: (B, 24) 或 (24,) → 每个方案的筛选指标。"""
        y = self.trajectories(theta)
        oil = np.einsum("bwt,t->b", y[:, :, 0, :], WT)          # 累计产油(梯形)
        oil3 = np.einsum("bwt,t->b", y[:, :, 0, :][:, :, MS3], WT[MS3])
        wat = np.einsum("bwt,t->b", y[:, :, 1, :], WT)          # 累计产水
        return {"oil": oil, "oil_3y": oil3, "water_prod": wat}


def _selftest() -> None:
    p = Proxy()
    t0 = time.time(); base = p.evaluate(np.zeros(24)); t1 = time.time()
    B = 2000
    rng = np.random.default_rng(0)
    t2 = time.time(); r = p.evaluate(rng.normal(0, .3, (B, 24)).astype(np.float32)); t3 = time.time()
    print(f"基准 θ=0  预测累计产油 = {base['oil'][0]:,.0f}  前3年 {base['oil_3y'][0]:,.0f}")
    print(f"单次评估 {1e3*(t1-t0):.2f} ms   批量 {B} 个耗时 {1e3*(t3-t2):.0f} ms "
          f"= {1e6*(t3-t2)/B:.1f} µs/方案")
    print(f"批量结果 产油范围 {r['oil'].min():,.0f} ~ {r['oil'].max():,.0f}")
    print(f"\n对照:OPM Flow 单次模拟约 60 s → 加速约 {60/((t3-t2)/B):,.0f} 倍")


if __name__ == "__main__":
    _selftest()
