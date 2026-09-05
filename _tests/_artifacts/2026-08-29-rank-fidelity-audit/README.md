# 对抗性复核证据 · fc_rank_fidelity (2026-08-29)

13 次**新** OPM Flow 模拟 + 全量重算脚本。审计者独立产出，不属于主流程。

- `audit_long0/long4/short6.npz` — 用 rf_long_0_c5 / rf_long_4_c4 / rf_short_6_c16 的 θ
  重跑 OPM Flow，与缓存 npz **逐位相同**（obs/inj_actual/field_cum maxdiff=0）→ 裁判确为模拟器。
- `audit_s9_*.npz`(4)、`audit_s6_*.npz`(6) — 在报告声称「±0.05% 不可达」的两个 s 区间内加密探测。
- `verif*.py` — 从 npz 原始 obs/inj_actual/field_cum 独立重算全部指标的脚本。

🔴 主要结论：注水量守恒用了『40 点瞬时率梯形积分』(fc_agent._metrics)，
而项目自己的 fc_tight（早 1.5 小时定版）已证明该口径对总注水量有 ±0.5% 残差、
必须改用模拟器自报累计 FWIT。按 FWIT 复核，10 个 short 候选**全部超出** ±0.05%
（最差 −1.39%），且所谓「阶梯函数」在 FWIT 上完全不存在。
