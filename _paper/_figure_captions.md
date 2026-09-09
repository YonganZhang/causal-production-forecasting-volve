# 图注

按 `share-sci-plot` 硬不变量 3:图内不放标题,解释性结论与精确结果放这里。
数字由 `python _code/fc_paper_check.py` 守住,不手抄。

---

**Fig. 1.** The Gaia agent team. *(整页示意图,走 `pptx-gen` / FAL NanoBanana2 生成,
spec 见 `_paper/_fig1_spec.json`。)*
**a**, Five layers and their agents, with the three levels of decision authority
distinguished: components that are fast but may not price (dashed), components that
reason but may not veto (solid), and the full-physics simulator, which alone assigns
value (heavy). Measured outcomes are fed back to every role. **b**, How each role's
knowledge base is assembled — a deck audit that records what the simulation model
does *not* contain, field-specific literature supplied as titles and DOIs only, and a
domain corpus distilled into general waterflood principles — together with the rule
separating admissible knowledge from knowledge that could only be written after
solving the problem. **c**, One decision round. **d**, The measured basis for the
division of authority.

**Fig. 2.** The surrogate: what it is, how it compares, and why the simulator still
adjudicates. **a**, The learned operator maps a static control vector directly to
per-well oil and water trajectories; no historical sequence is supplied at inference.
**b**, Field-level relative error of seven models fitted under one protocol (n = 2,000;
400 held out). The final ensemble reaches 0.478 % (R² = 0.9926) against 3.043 % for a
random forest and 5.745 % for a trivial mean predictor; the dotted line marks the
0.047 % structural floor obtained by projecting the truth onto its own leading
principal components. **c**, Predicted and simulated rates for the two highest-rate
producers. **d**, Rank fidelity against the simulator in two regimes. On 500 randomly
sampled schedules the surrogate reproduces the simulator's ordering (ρ = 0.998,
top-1 hit rate 1.0); on the ten schedules an optimiser produced by searching against
that same surrogate, the ordering carries no usable signal (ρ = −0.21, top-1 hit rate 0)
and its first choice is the simulator's last.

**Fig. 3.** How the team reaches a decision. **a**, Size of the evidence slice held by
each role in a representative round, coloured by provenance type; two of seven roles
hold no field-specific measurement. **b**, Candidates per round entering, surviving
surrogate screening and the arithmetic feasibility gate, and being adjudicated by the
full-physics simulator (30 rounds over 10 runs). The gate is arithmetic and is never
delegated to a language model. **c**, Front-loading index τ of accepted schedules when
the synthesiser weights assertions by head count versus by evidence strength.

**Fig. 4.** Economic outcome and its robustness. **a**, ΔNPV at 8 % over the historical
schedule for each repetition, against the one-dimensional scaling reference.
**b**, Physical change delivered by the best adjudicated schedule: cumulative oil rises
10.9 % while injected water falls 6.5 % and produced water rises 1.7 %, so the gain comes
from redistributing injection in time rather than from adding volume. **c**, The same
schedule remains profitable at every discount rate tested.

**Fig. 5.** Comparison and decision efficiency. **a**, Mean gain with one standard
deviation. **b**, Full-physics evaluations consumed per run. **c**, Best result so far
within each run; every run finishes at or above where it started, and above the
reference strategy.

**Fig. 6.** What the framework recommends, and why it is legible. **a**, The best
adjudicated schedule as base-10 rate multipliers over six control periods and four
injectors; it resolves into three blocks and leaves the weakest injector at its baseline
rate. **b**, ΔNPV against the front-loading index τ on 275 held-out random cases, with
all agent- and optimiser-generated solutions excluded; the dashed line marks the mean τ
of accepted schedules. **c**, Estimated injector influence on cumulative oil from partial
regression over 471 simulated cases (design-matrix condition number 2.05).

---

## 从图内移出的文字(硬不变量 3)

原来写在图上的这些标题已删,内容并入上面的图注:
`Gaia: five layers…` / `How role knowledge is built` / `One decision round` /
`Why authority is split this way` / `Operator learning, not forecasting` /
`Benchmark under one protocol` / `Predicted vs simulated well curves` /
`Why physics still adjudicates` / `Evidence held per role` / `Candidate funnel` /
`Adjudication rule` / `Simulator-verified gain` / `What changed physically` /
`Robust across capital cost` / `Gain over historical schedule` /
`Cost of the decision` / `Closed-loop progress` / `Recommended schedule` /
`One readable direction` / `Which knobs are real`
