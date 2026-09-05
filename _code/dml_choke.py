"""P1 原型:油嘴开度对产液/产油速率的因果剂量-响应(DML + 时间块交叉拟合)。

来源:文献 Workflow 综合出的候选问题 P1(见 _refs/_index.md)。本文件是主会话
**独立复核** 子智能体自报数字的实现,不是照抄其原型。

方法(Chernozhukov et al. 2018 的偏线性模型 + Robinson partialling-out):
    q_it = θ_i · D_it + g_i(X_it) + U_it
    D_it = m_i(X_it) + V_it
    θ̂_i = Σ r^D r^Y / Σ (r^D)²,   r^D = D - m̂(X), r^Y = q - ℓ̂(X)

关键改动(相对原文):原文假设 i.i.d.、用随机 K 折。生产数据强自相关,随机折会让
辅助折泄漏主折信息。这里用**连续时间块** + embargo,并同时报随机折结果以量化泄漏。

口径纪律:
  - 目标是**速率**(体积/开井小时),不是体积 —— 结构上免疫记账恒等式。
  - 关井日速率无定义 → 不进估计样本,但保留为滞后状态变量。
  - 缺日显式补 NaN,滞后特征跨缺日整行丢弃,绝不 ffill。
  - X 全部是滞后量,不含任何同期或未来信息。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from volve_data import PRODUCERS, calendarize_well, load_daily

LAGS = (1, 2, 3, 7)
# 🔴 2026-08-07 审计:速率 = 体积/开井小时, 分母极小的日子会产生巨大杠杆点
# (实测开井 0.625 小时 -> qo=1012.7, 而正常上限约 70)。这些点原样进入估计并主导 θ。
MIN_ON_HOURS = 4.0
BASE_VARS = ("qL", "qo", "fw", "AVG_DOWNHOLE_PRESSURE", "AVG_WHP_P", "AVG_WHT_P",
             "AVG_CHOKE_SIZE_P")
TREAT = "AVG_CHOKE_SIZE_P"
EPS = 1e-9


def build_well_frame(daily: pd.DataFrame, code: str) -> pd.DataFrame:
    """单井日历网格 + 速率口径 + 滞后特征。"""
    g = calendarize_well(daily[daily["WELL_BORE_CODE"] == code])
    hrs = g["ON_STREAM_HRS"]
    open_ = hrs >= MIN_ON_HOURS
    liq = g["BORE_OIL_VOL"] + g["BORE_WAT_VOL"]
    g["qL"] = np.where(open_, liq / hrs.where(open_), np.nan)
    g["qo"] = np.where(open_, g["BORE_OIL_VOL"] / hrs.where(open_), np.nan)
    g["fw"] = np.where(liq > EPS, g["BORE_WAT_VOL"] / liq.where(liq > EPS), np.nan)
    g["shut_in"] = (~open_ & hrs.notna()).astype(float)
    # 🔴 2026-08-07 审计:旧版把当期 cum_oil / t_idx 放进特征。cum_oil[t] 精确包含当日
    # BORE_OIL_VOL[t], 而 target=qo 时 Y[t] 正是它除以小时数 —— 直接泄漏; 且二者都是
    # 近似唯一的行标识(cum_oil 唯一值比例 0.903), 在随机折下是记忆化通道。
    # 改为滞后 1 天的累计产油, 并删掉 t_idx(块折下它只能外推, 没有信息价值)。
    g["cum_oil_lag1"] = g["BORE_OIL_VOL"].fillna(0).cumsum().shift(1)

    for v in BASE_VARS + ("shut_in",):
        for L in LAGS:
            g[f"{v}_lag{L}"] = g[v].shift(L)
    return g


def design(g: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    feats = [f"{v}_lag{L}" for v in BASE_VARS + ("shut_in",) for L in LAGS] + ["cum_oil_lag1"]
    need = feats + [target, TREAT, "ON_STREAM_HRS"]
    sub = g.loc[:, need].copy()
    sub = sub[g["ON_STREAM_HRS"] >= MIN_ON_HOURS]   # 关井日与极短开井日不进估计样本
    sub = sub.dropna()                            # 缺日/缺值整行丢弃, 不 ffill
    return sub[feats].to_numpy(), sub[TREAT].to_numpy(), sub[target].to_numpy(), sub.index


def _folds(index: pd.DatetimeIndex, k: int, embargo_days: int, block: bool, seed: int = 2026):
    n = len(index)
    if block:
        edges = np.linspace(0, n, k + 1).astype(int)
        for j in range(k):
            te = np.zeros(n, bool); te[edges[j]:edges[j + 1]] = True
            lo = index[edges[j]] - pd.Timedelta(days=embargo_days)
            hi = index[edges[j + 1] - 1] + pd.Timedelta(days=embargo_days)
            tr = ~te & ~((index >= lo) & (index <= hi))
            if tr.sum() > 50:
                yield tr, te
    else:
        rng = np.random.default_rng(seed)
        assign = rng.integers(0, k, n)
        for j in range(k):
            te = assign == j
            yield ~te, te


def dml_theta(X, D, Y, index, *, k=5, embargo_days=20, block=True, seed=2026):
    """返回 (θ̂, SE, R²(D|X), n)。nuisance 用梯度提升树。"""
    from sklearn.ensemble import HistGradientBoostingRegressor as GBR

    rD = np.full(len(D), np.nan); rY = np.full(len(Y), np.nan)
    for tr, te in _folds(index, k, embargo_days, block, seed):
        mk = GBR(max_iter=300, learning_rate=0.05, max_depth=4, random_state=seed).fit(X[tr], D[tr])
        lk = GBR(max_iter=300, learning_rate=0.05, max_depth=4, random_state=seed).fit(X[tr], Y[tr])
        rD[te] = D[te] - mk.predict(X[te])
        rY[te] = Y[te] - lk.predict(X[te])
    m = np.isfinite(rD) & np.isfinite(rY)
    rD, rY = rD[m], rY[m]
    denom = float((rD ** 2).sum())
    theta = float((rD * rY).sum() / denom)
    resid = rY - theta * rD
    se = float(np.sqrt(((rD ** 2) * (resid ** 2)).sum()) / denom)     # HC 型
    r2 = float(1 - rD.var() / D[m].var())
    return theta, se, r2, int(m.sum())


def residualize(X, D, index, *, k=5, embargo_days=20, block=True, seed=2026):
    """把处理变量 D 拆成「制度可解释的部分」与「意外变动」。返回残差与 R²。"""
    from sklearn.ensemble import HistGradientBoostingRegressor as GBR

    r = np.full(len(D), np.nan)
    for tr, te in _folds(index, k, embargo_days, block, seed):
        m = GBR(max_iter=300, learning_rate=0.05, max_depth=4, random_state=seed).fit(X[tr], D[tr])
        r[te] = D[te] - m.predict(X[te])
    ok = np.isfinite(r)
    return r, float(1 - r[ok].var() / D[ok].var())


def placebo_test(g: pd.DataFrame, target: str = "qo", shifts=(-1, 0, 1, 3, 7), **kw) -> dict:
    """安慰剂检验 —— 判断一口井的因果效应是否**可识别**。

    因果只能向前:今天的「意外调嘴」不可能影响 s 天前的产量。把结果变量往过去平移 s 天
    后重估 θ,若仍显著非零,说明残差里还留着未观测混杂。

    🔴 2026-08-07 审计发现的致命实现错误(已修):
        旧版只平移 y、不平移 X。而 LAGS=(1,2,3,7) 恰好覆盖了安慰剂位移 (1,3,7),
        于是平移后的 y **就是 X 里的 qo_lag{s} 那一列**(实测完全相同),
        nuisance 模型对它 R²=0.9999,残差被压到几乎为零,θ_placebo 机械趋近 0
        —— 与有没有混杂完全无关。整个可识别性审计因此是空的。

    正确做法:把**整个设计矩阵一起平移** s 天。这样问的是
    「以 t-s 时刻**之前**的信息为条件,t 时刻的意外调嘴能否解释 t-s 时刻的产量」,
    条件集与被解释变量之间不再有恒等关系。
    """
    from sklearn.ensemble import HistGradientBoostingRegressor as GBR

    X0, D0, _, idx0 = design(g, target)
    if len(D0) < 200:
        return {}
    pos = {ts: i for i, ts in enumerate(idx0)}
    out = {}
    for sh in shifts:
        # 平移后仍在样本内的行:t 与 t-sh 都必须存在
        pairs = [(pos[ts], pos[ts - pd.Timedelta(days=sh)])
                 for ts in idx0 if (ts - pd.Timedelta(days=sh)) in pos]
        if len(pairs) < 200:
            out[sh] = float("nan")
            continue
        cur = np.array([a for a, _ in pairs])      # 处理变量所在的时刻 t
        lag = np.array([b for _, b in pairs])      # 结果变量所在的时刻 t-sh
        Xs, Ds, Ys = X0[lag], D0[cur], design(g, target)[2][lag]
        ids = idx0[lag]
        rD = np.full(len(Ds), np.nan); rY = np.full(len(Ys), np.nan)
        for tr, te in _folds(ids, kw.get("k", 5), kw.get("embargo_days", 20), True):
            mk = GBR(max_iter=300, learning_rate=0.05, max_depth=4,
                     random_state=2026).fit(Xs[tr], Ds[tr])
            lk = GBR(max_iter=300, learning_rate=0.05, max_depth=4,
                     random_state=2026).fit(Xs[tr], Ys[tr])
            rD[te] = Ds[te] - mk.predict(Xs[te])
            rY[te] = Ys[te] - lk.predict(Xs[te])
        m = np.isfinite(rD) & np.isfinite(rY)
        out[sh] = float((rD[m] * rY[m]).sum() / (rD[m] ** 2).sum()) if m.sum() > 50 else float("nan")
    return out


def naive_slope(D, Y) -> float:
    """无任何控制的 OLS 斜率。"""
    A = np.vstack([D, np.ones_like(D)]).T
    return float(np.linalg.lstsq(A, Y, rcond=None)[0][0])


def run(target: str = "qL", block: bool = True) -> pd.DataFrame:
    daily = load_daily()
    rows, pool_D, pool_Y = [], [], []
    for code, short in PRODUCERS.items():
        g = build_well_frame(daily, code)
        X, D, Y, idx = design(g, target)
        if len(D) < 200:
            rows.append({"well": short, "n": len(D), "note": "样本不足"})
            continue
        pool_D.append(D); pool_Y.append(Y)
        th, se, r2, n = dml_theta(X, D, Y, idx, block=block)
        rows.append({"well": short, "n": n, "naive": naive_slope(D, Y),
                     "dml": th, "se": se, "r2_D_given_X": r2,
                     "sign_ok": th >= 0})
    tab = pd.DataFrame(rows)
    tab.attrs["pooled_naive"] = naive_slope(np.concatenate(pool_D), np.concatenate(pool_Y))
    tab.attrs["n_total"] = int(sum(len(x) for x in pool_D))
    return tab


if __name__ == "__main__":
    for target in ("qL", "qo"):
        tab = run(target)
        print(f"\n===== 目标 = {target}（{'产液速率' if target=='qL' else '产油速率'} Sm³/开井小时）=====")
        print(f"合并朴素斜率 = {tab.attrs['pooled_naive']:+.4f}   有效样本合计 n = {tab.attrs['n_total']}")
        print(tab.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
        if "sign_ok" in tab:
            print(f"物理符号约束 ∂q/∂choke ≥ 0：DML {int(tab.sign_ok.sum())}/{len(tab)} 通过，"
                  f"朴素 {int((tab.naive >= 0).sum())}/{len(tab)} 通过")

    print("\n===== 泄漏量化：块交叉拟合 vs 随机 K 折（目标 qL）=====")
    b, r = run("qL", block=True), run("qL", block=False)
    cmp = b[["well", "n", "dml"]].merge(r[["well", "dml"]], on="well", suffixes=("_block", "_random"))
    cmp["diff"] = cmp.dml_random - cmp.dml_block
    print(cmp.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))


def identifiability_audit(target: str = "qo") -> pd.DataFrame:
    """逐井可识别性审计:制度可解释比例 + 安慰剂检验。

    这是本项目对文献的主要方法学补充 —— 精读的 16 篇里没有任何一篇做过。
    判定规则:安慰剂 θ 的绝对值 >= 当日 θ 的一半 => 该井不可识别, 其估计值不得引用。
    """
    daily = load_daily()
    rows = []
    for code, short in PRODUCERS.items():
        g = build_well_frame(daily, code)
        X, D, Y, idx = design(g, target)
        if len(D) < 200:
            continue
        _, r2 = residualize(X, D, idx)
        pb = placebo_test(g, target)
        eff, plac = pb.get(0, np.nan), max(abs(pb.get(s, 0.0)) for s in (1, 3, 7))
        # 🔴 nuisance 失败门:R2(D|X) <= 0 表示处理方程模型比常数均值还差,
        # 此时 rD 已不是"意外调嘴"而是被放大的噪声, θ 不可解释, 一律判不可识别。
        nuisance_ok = np.isfinite(r2) and r2 > 0.05
        rows.append({"well": short, "n": len(D), "policy_r2": r2,
                     "theta_now": eff, "placebo_max": plac,
                     "nuisance_ok": nuisance_ok,
                     "identifiable": bool(nuisance_ok and abs(eff) > 2 * plac)})
    return pd.DataFrame(rows)
