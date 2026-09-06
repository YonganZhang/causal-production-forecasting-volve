# 决策:投 Petroleum Science,主线改为「模拟器保留否决权」

日期 2026-09-06。依据:网页咨询(ChatGPT,对话 `投稿策略评估`)+ 本地独立核验。

## 目标期刊

Petroleum Science(KeAi/Elsevier,ISSN 1995-8226)。**已核实**:IF 6.4、
CiteScore 11.3(2025 JCR)。近期确有油藏 ML 论文,如
"Knowledge transfer across development stages for dynamic reservoir production
optimization"、"Deep learning-based upscaling for reservoir models on
corner-point grids"。**未检索到 LLM/智能体类论文** —— 既是新颖性也是风险:
审稿人可能不熟悉这条技术线,方法必须自证而不能假设读者懂 LLM。

## 主线(改)

不写「七个 LLM 专家把 Norne 优化好了」,写:

> 代理模型提供规模,全物理模拟器保留否决权。AI 可以提出与筛选决策,
> 但每一个被接受的方案都由 OPM Flow 裁定,工程真实性不托付给 AI。

Norne 是 demonstration,不是受建议方。

## 🔴 2026-09-06 晚:用户裁定覆盖本节

**唯一的无 Agent 对照是一维缩放。B4 等预算随机、BG 贪心短期已退役。**
下面这一节记录的"把贪心升为正文主图"是我的建议，**已被用户否决**，
不再是本项目的执行方案。退役理由、我提出的反对意见与恢复方式见
`_legacy/2026-09-06-baselines-B4-BG-retired/README.md`。

保留本节的原因:审稿人若追问强基线，这里有现成的数字可答。

## （已否决）咨询提出、经本地核验后**推翻**的一条

咨询把「统计显著性只来自最弱基线」列为头号拒稿风险。**核验结果相反**:

| 对照 | Agent Team 差值 | 95% CI | p |
|---|---|---|---|
| 一维缩放 (7 次模拟) | +298.6M | [+262.4, +334.9] | 1.70e-08 |
| 等预算随机 (3 次) | +114.3M | [+78.0, +150.5] | 5.53e-05 |
| **贪心短期 (31 次)** | **+67.7M** | **[+31.5, +104.0]** | **2.23e-03** |

**对最强基线仍显著**,且智能体每次实验只用 **2.7 次** OPM Flow 调用,
贪心用 31 次 —— **11 倍模拟器预算优势**。这不是软肋,是本文最强卖点,
应升为正文主图:ΔNPV vs 模拟器调用次数。

## 咨询提出、经本地核验后**采纳**的

1. **不能说多角色提升稳健性**。Levene p=0.453、Bartlett p=0.295,
   方差差异不可检验。sd 50.7 vs 72.9 只能写 descriptive dispersion。
2. **不能说 causal law / 可迁移规律**。tilt 只有 Spearman 0.714,
   无干预识别假设。降格为 interpretable scheduling tendency,
   作用是证明智能体优解不是黑箱随机,而非提出油藏新知。
3. **不能说逼近专家水平** —— 无人类专家基线。
4. **p=0.63 ≠ 两法等价**。Team−单Agent 95% CI [−45.2, +72.8](Cohen d=0.22),
   只能写 no detectable difference。
5. Results 顺序:代理精度 → 经济表现全基线同图 → 多角色消融(negative)。
   negative result 的定位是「消融定位了真正起作用的机制」,不是「我们的方法没用」。

## 待办(已按用户裁定修订)

- P0 正文主图 ΔNPV vs 模拟器预算 —— **仍然做**,但对照只放一维缩放。
  即便如此仍有说服力:Agent 每次实验用 **2.7 次**真模拟,一维缩放用 **7 次**。
- P1 代理模型补 top-k 排名保真度(全局 R² 不等于优化区可用)
- P1 Norne 内做经济假设/初始方案的 robustness test,替代第二个油藏
