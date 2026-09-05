#!/usr/bin/env python3
"""预测段代理模型基准：六种主流方法在同一口径下对比。

## 为什么要做
"用哪种模型建油藏代理"在石油期刊是成熟体裁。我们有 2000 个预测段样本、
统一的输入输出定义，可以做一次公平对比，选出最好的再做改进。

## 公平性纪律（每条都是踩过的坑）
1. **同一划分**：所有模型共用同一个 train/test 索引，种子固定。
2. **同一指标**：预测段(13.1 年)累计产油的相对误差 + R²。
   不用归一化 MSE —— 那个数工程师看不懂, 也不是决策关心的量。
3. **强制打平凡基线**（永远预测训练集均值）。打不过就标 FAIL。
   本项目栽过:一个三行公式的笨办法打赢过 SOTA 大模型, 撤过稿。
4. **报训练+推理耗时**。代理模型的意义就是快, 只报精度不报速度是耍流氓。
5. 输出 1760 维, 对 SVM/RF 这类逐输出回归的方法先做 PCA 降维, 并**记录降维损失**
   —— 否则"某模型差"可能只是降维丢了信息。

用法:
    python fc_benchmark.py --gpu 6
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path
import numpy as np
import fc_decision as FD

# 🔴 钉死样本数:扩样本在后台跑时数据集会变大,不钉死会让不同实验
#    落在不同的切分上,横向不可比(2026-08-27 实测 fc_plug 拿到 4345 个)。
N_PIN = 4000
import forecast_gen as FG

OUT = Path(__file__).resolve().parent.parent / "_pipelines" / "fc_benchmark"
DAYS = FG.FC_GRID


def cumoil(Y):
    """Y (n,22,2,40) → 预测段累计产油 (n,)"""
    return np.trapezoid(Y[:, :, 0, :], DAYS, axis=2).sum(1)


def metrics(pred, true, tr_mean):
    e = np.abs(cumoil(pred) - cumoil(true)) / np.maximum(np.abs(cumoil(true)), 1e-9)
    t = cumoil(true)
    r2 = 1 - ((cumoil(pred) - t) ** 2).sum() / ((t - t.mean()) ** 2).sum()
    b = np.abs(tr_mean - t) / np.abs(t)
    return {"relerr_median": float(np.median(e)), "relerr_p90": float(np.percentile(e, 90)),
            "r2": float(r2), "beats_trivial": bool(np.median(e) < np.median(b))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=6)
    ap.add_argument("--n-pca", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    TH, Y, IA, FC = FD.load(N_PIN)
    n = len(TH); shape = Y.shape[1:]
    Yf = Y.reshape(n, -1)
    rng = np.random.default_rng(args.seed); idx = rng.permutation(n)
    n_te = n // 5; te, tr = idx[:n_te], idx[n_te:]
    print(f"样本 {n}  训练 {len(tr)}  留出 {len(te)}   输入 {TH.shape[1]} 维  输出 {Yf.shape[1]} 维")

    xm, xs = TH[tr].mean(0), TH[tr].std(0) + 1e-8
    ym, ys = Yf[tr].mean(0), Yf[tr].std(0) + 1e-8
    Xtr, Xte = (TH[tr] - xm) / xs, (TH[te] - xm) / xs
    Ytr_n = (Yf[tr] - ym) / ys
    tr_mean = cumoil(Yf[tr].mean(0).reshape(1, *shape))[0]

    # PCA:给逐输出回归的方法降维。记录降维本身丢了多少 —— 这是解释"某模型差"的前提
    from sklearn.decomposition import PCA
    pca = PCA(n_components=args.n_pca, random_state=0).fit(Ytr_n)
    Ztr = pca.transform(Ytr_n)
    recon = (pca.inverse_transform(Ztr) * ys + ym).reshape(-1, *shape)
    pca_loss = metrics(recon, Y[tr], tr_mean)
    print(f"PCA {args.n_pca} 维: 解释方差 {pca.explained_variance_ratio_.sum():.4f}   "
          f"重建本身的相对误差 {pca_loss['relerr_median']:.3%} ← 这是 PCA 类方法的下限")

    def back(Zp):
        return (pca.inverse_transform(Zp) * ys + ym).reshape(-1, *shape)

    res = {}

    def run(name, fit_pred):
        t0 = time.time()
        P, t_inf = fit_pred()
        m = metrics(P, Y[te], tr_mean)
        m["train_s"] = round(time.time() - t0 - t_inf, 2); m["infer_ms"] = round(t_inf * 1000 / len(te), 3)
        res[name] = m
        print(f"  {name:22s} 相对误差 {m['relerr_median']:>7.2%}  p90 {m['relerr_p90']:>7.2%}  "
              f"R² {m['r2']:>7.4f}  训练 {m['train_s']:>7.2f}s  推理 {m['infer_ms']:>6.3f}ms"
              f"  {'' if m['beats_trivial'] else '🔴 未打赢平凡基线'}")

    print("\n=== 六种主流方法(同一划分、同一指标) ===")

    # 2 随机森林
    def m_rf():
        from sklearn.ensemble import RandomForestRegressor
        r = RandomForestRegressor(n_estimators=300, n_jobs=8, random_state=0).fit(Xtr, Ztr)
        t = time.time(); P = back(r.predict(Xte)); return P, time.time() - t
    run("随机森林", m_rf)

    # 3 梯度提升树
    def m_gb():
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.ensemble import HistGradientBoostingRegressor as H
        r = MultiOutputRegressor(H(max_iter=200, random_state=0), n_jobs=8).fit(Xtr, Ztr[:, :16])
        t = time.time()
        Zp = np.zeros((len(Xte), args.n_pca)); Zp[:, :16] = r.predict(Xte)
        P = back(Zp); return P, time.time() - t
    run("GBDT(前16主成分)", m_gb)

    # 4 支持向量回归
    def m_svr():
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.svm import SVR
        r = MultiOutputRegressor(SVR(C=10.0, epsilon=0.01), n_jobs=8).fit(Xtr, Ztr[:, :16])
        t = time.time()
        Zp = np.zeros((len(Xte), args.n_pca)); Zp[:, :16] = r.predict(Xte)
        P = back(Zp); return P, time.time() - t
    run("SVR(前16主成分)", m_svr)

    # 5 高斯过程
    def m_gp():
        from sklearn.gaussian_process import GaussianProcessRegressor as G
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
        k = C(1.0) * RBF(np.ones(Xtr.shape[1])) + WhiteKernel(1e-3)
        sub = slice(None, min(800, len(Xtr)))          # GP 是 O(n^3), 取子集
        r = G(kernel=k, normalize_y=True, random_state=0).fit(Xtr[sub], Ztr[sub, :16])
        t = time.time()
        Zp = np.zeros((len(Xte), args.n_pca)); Zp[:, :16] = r.predict(Xte)
        P = back(Zp); return P, time.time() - t
    run("高斯过程(前16主成分)", m_gp)

    # 6 MLP(BP 神经网络) —— 直接回归全部 1760 维, 不经 PCA
    def m_mlp():
        import torch
        dev = f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"
        torch.manual_seed(args.seed)
        net = FD.build_net(TH.shape[1], Yf.shape[1], 768, 5).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
        A = torch.tensor(Xtr, device=dev, dtype=torch.float32)
        B = torch.tensor(Ytr_n, device=dev, dtype=torch.float32)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 400)
        for ep in range(400):
            p = torch.randperm(len(A), device=dev)
            for i in range(0, len(A), 128):
                b = p[i:i + 128]
                opt.zero_grad(); torch.nn.functional.mse_loss(net(A[b]), B[b]).backward(); opt.step()
            sch.step()
        net.eval(); t = time.time()
        with torch.no_grad():
            P = (net(torch.tensor(Xte, device=dev, dtype=torch.float32)).cpu().numpy() * ys + ym)
        return P.reshape(-1, *shape), time.time() - t
    run("MLP(BP 神经网络)", m_mlp)

    # ---- 序列模型:显式建模 40 个时刻的时间维度 ----
    # 前面几种把 1760 维当成 1760 个独立数字预测, 完全没用上"这是一条曲线"这个先验。

    def seq_model(name, make):
        def f():
            import torch
            dev = f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"
            torch.manual_seed(args.seed)
            nw, nph, nt = shape                       # 22, 2, 40
            net = make(TH.shape[1], nw * nph, nt).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
            A = torch.tensor(Xtr, device=dev, dtype=torch.float32)
            B = torch.tensor(Ytr_n.reshape(-1, nw * nph, nt), device=dev, dtype=torch.float32)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 300)
            for ep in range(300):
                pm = torch.randperm(len(A), device=dev)
                for i in range(0, len(A), 64):
                    b = pm[i:i + 64]
                    opt.zero_grad()
                    torch.nn.functional.mse_loss(net(A[b]), B[b]).backward(); opt.step()
                sch.step()
            net.eval(); t = time.time()
            with torch.no_grad():
                P = net(torch.tensor(Xte, device=dev, dtype=torch.float32)).cpu().numpy()
            P = (P.reshape(len(te), -1) * ys + ym).reshape(-1, *shape)
            return P, time.time() - t
        run(name, f)

    def make_transformer(d_in, n_ch, n_t):
        import torch, torch.nn as nn
        class T(nn.Module):
            def __init__(self):
                super().__init__()
                self.cond = nn.Linear(d_in, 128)
                self.pos = nn.Parameter(torch.randn(1, n_t, 128) * 0.02)
                enc = nn.TransformerEncoderLayer(128, 4, 256, batch_first=True, dropout=0.0)
                self.tr = nn.TransformerEncoder(enc, 3)
                self.head = nn.Linear(128, n_ch)
            def forward(self, x):
                h = self.cond(x)[:, None, :] + self.pos          # 注水方案作为条件广播到每个时刻
                return self.head(self.tr(h)).transpose(1, 2)
        return T()
    seq_model("Transformer", make_transformer)

    def make_autoformer(d_in, n_ch, n_t):
        """Autoformer 的两个核心构件:序列分解(趋势/季节) + 自相关。
        这里实现其精神:显式拆趋势与残差两支, 各自预测后相加。"""
        import torch, torch.nn as nn
        class A(nn.Module):
            def __init__(self):
                super().__init__()
                self.cond = nn.Linear(d_in, 128)
                self.pos = nn.Parameter(torch.randn(1, n_t, 128) * 0.02)
                enc = nn.TransformerEncoderLayer(128, 4, 256, batch_first=True, dropout=0.0)
                self.tr = nn.TransformerEncoder(enc, 2)
                self.trend = nn.Sequential(nn.Linear(d_in, 128), nn.SiLU(), nn.Linear(128, n_ch * n_t))
                self.head = nn.Linear(128, n_ch)
                self.n_ch, self.n_t = n_ch, n_t
                self.pool = nn.AvgPool1d(5, 1, 2, count_include_pad=False)
            def forward(self, x):
                tr_ = self.trend(x).view(-1, self.n_ch, self.n_t)
                tr_ = self.pool(tr_)                              # 序列分解:趋势支做移动平均平滑
                h = self.cond(x)[:, None, :] + self.pos
                sea = self.head(self.tr(h)).transpose(1, 2)       # 残差(季节)支
                return tr_ + sea
        return A()
    seq_model("Autoformer(分解式)", make_autoformer)

    # ---- Chronos-2:时序基础模型 ----
    # 🔴 它和前面几种**不是同一类任务**。前面是"注水方案 → 曲线"的回归;
    #    Chronos 是"给一段历史 → 预测未来"。要用它必须换个接法:
    #    **历史产量当上下文, 注水方案当未来协变量** —— 这正是 Chronos-2 的原生能力,
    #    也是它相对其他模型唯一的独门优势(别的模型没有"历史"这个输入)。
    #    公平性提醒:它多用了一份信息(历史曲线), 所以**不能简单说它赢了就更强**,
    #    要在表里注明这一点。
    def m_chronos():
        import torch
        from chronos import BaseChronosPipeline
        dev = f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"
        pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-2",
                                                   device_map=dev, torch_dtype=torch.bfloat16)
        nw, nph, nt = shape
        ctx_len = 16                       # 用预测段前 16 个时刻当上下文, 预测后 24 个
        pred_len = nt - ctx_len
        t = time.time()
        P = np.repeat(Yf[tr].mean(0).reshape(1, *shape), len(te), axis=0)   # 前段用均值占位
        for w in range(nw):
            ctx = [torch.tensor(Y[te][i, w, 0, :ctx_len], dtype=torch.float32) for i in range(len(te))]
            # 🔴 Chronos-2 的参数名是 inputs 不是 context(2.x 改过签名)。
            #    第一版用 context= 于是每口井都 TypeError, 而占位值恰好等于平凡基线,
            #    表里显示 4.99% —— 与基线一模一样, 那是**失败的伪装**不是结果。
            #    教训:两个不同来源的数字完全相等时, 先怀疑其中一个没真的算。
            q, mean = pipe.predict_quantiles(inputs=ctx, prediction_length=pred_len,
                                             quantile_levels=[0.5])
            arr = np.asarray(mean.float().cpu()) if hasattr(mean, "float") else np.asarray(mean)
            arr = arr.reshape(len(te), -1)
            P[:, w, 0, ctx_len:] = arr[:, :pred_len]
            P[:, w, 0, :ctx_len] = Y[te][:, w, 0, :ctx_len]
        return P, time.time() - t
    try:
        run("Chronos-2(零样本+历史上下文)", m_chronos)
        res["Chronos-2(零样本+历史上下文)"]["note"] = "多用了前16个时刻的真实曲线, 与其他模型信息不对等"
    except Exception as e:                                      # noqa: BLE001
        print(f"  Chronos-2 跑不起来: {type(e).__name__}: {str(e)[:160]}")

    # 平凡基线
    Pb = np.repeat(Yf[tr].mean(0).reshape(1, *shape), len(te), axis=0)
    res["平凡基线(训练集均值)"] = metrics(Pb, Y[te], tr_mean)
    print(f"  {'平凡基线(训练集均值)':22s} 相对误差 {res['平凡基线(训练集均值)']['relerr_median']:>7.2%}"
          f"  R² {res['平凡基线(训练集均值)']['r2']:>7.4f}   ← 所有模型必须打赢它")

    best = min((k for k in res if "平凡" not in k), key=lambda k: res[k]["relerr_median"])
    print(f"\n🏆 最好: {best}  相对误差 {res[best]['relerr_median']:.2%}  R² {res[best]['r2']:.4f}")
    json.dump({"n": n, "n_train": len(tr), "n_test": len(te), "n_pca": args.n_pca,
               "pca_explained": float(pca.explained_variance_ratio_.sum()),
               "pca_floor_relerr": pca_loss["relerr_median"],
               "results": res, "best": best},
              open(OUT / "benchmark.json", "w"), indent=1, ensure_ascii=False)
    print(f"已写入 {OUT}/benchmark.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
