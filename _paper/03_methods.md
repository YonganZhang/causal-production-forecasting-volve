# 3. Methods

## 3.0 Overview

The framework couples three components into a closed loop (Fig. 1). A learned operator acts as a
fast stand-in for the reservoir simulator so that candidate schedules can be screened in bulk
(§3.2). A team of language-model agents, each restricted to a distinct evidence slice, proposes
and cross-examines candidate schedules (§3.4). Every surviving candidate is adjudicated by the
full-physics simulator, and the realised outcome is returned as the next round's input (§3.4).

The division of labour is deliberate and is justified empirically rather than by preference:
**the surrogate supplies scale, the simulator supplies trust.** §3.3 establishes why the second
half of that sentence is necessary — the same surrogate that ranks random schedules almost
perfectly ranks the optimiser's own shortlist no better than chance, and selects precisely the
candidate whose objective it overstates most. A framework that let the surrogate make the final
call would therefore be misled exactly where the decision is made.

## 3.1 Problem setting and economic objective

We consider waterflood control on the Norne field model (46 × 112 × 22 grid, 44,431 active cells,
four water injectors F-1H–F-4H and 22 producers). The decision variable is a piecewise-constant
injection schedule

  θ ∈ R^(S×I),  S = 6 control periods (2007, 2008, 2010, 2012, 2014, 2017),  I = 4 injectors,

where θ_{s,i} is the base-10 multiplier applied to injector *i*'s baseline rate during period *s*
over a 13-year forecast horizon. History is simulated once; every candidate branches from the same
restart state, so schedules differ only in the forecast segment.

The objective is net present value under an explicit water cost,

  NPV(r) = Σ_t [ q_o(t)·p_oil(t) − q_wi(t)·c_inj − q_wp(t)·c_prod ] · (1+r)^(−t),

with p_oil taken from EIA annual Brent spot prices, c_inj and c_prod the per-barrel costs of
injected and of produced water, and r the discount rate. Unless stated otherwise we report
r = 8 %, c_inj = 2.0 US$/bbl and c_prod = 1.0 US$/bbl. The historical schedule gives
NPV_8% = 1,773.4 M US$, which is the reference for every ΔNPV in this paper.

Pricing water is not cosmetic. Under oil revenue alone the question *how much water should be
injected* degenerates: water is free and monotonically increases oil, so the optimum sits on the
boundary. Earlier formulations therefore imposed a fixed injection budget — a hard constraint
standing in for a missing cost term. Charging for water makes total injection an **endogenous**
decision rather than a boundary artefact.

All cumulative quantities are read from the simulator's own running totals (FOPT, FWIT, FWPT)
rather than integrated from well curves. The two conventions disagree by 0.58 % on average in our
runs, one to two orders of magnitude above the tolerances used elsewhere in this work, and in 14
cases they returned opposite feasibility verdicts for the same schedule.

## 3.2 A learned operator as a fast stand-in for the simulator

**Motivation.** A full-physics run of this model takes approximately one minute. Any decision
procedure that requires large-scale trial and error is therefore impractical at the simulator's
own cost.

**Design.** We train a parametric operator mapping the static control vector directly to
multi-well, multi-phase production trajectories,

  f_φ : R^24 → R^(W×P×T),  W active producers, P ∈ {oil, water}, T = 40 time nodes.

This is operator learning rather than time-series forecasting: **no historical sequence is
supplied at inference time**. The architecture is a conditioning MLP followed by a Transformer
encoder over time nodes, with Fourier positional embeddings (16 bands) in place of a learnable
positional table. Training uses 7,400 simulated samples. A further 500 samples, generated
independently *after* model selection was frozen, serve as a confirmation set evaluated exactly
once.

**Performance.** On that confirmation set the field-level relative error is 0.369 % and
R² = 0.9959. Batched inference costs 18.1 µs per candidate, about 3.3 × 10⁶ times faster than the
corresponding simulation.

## 3.3 Where the surrogate fails, and why the simulator must adjudicate

An accurate surrogate is not automatically a trustworthy judge. We therefore measured its **rank**
fidelity separately in two regimes, using OPM Flow as the arbiter in both.

On 500 randomly sampled schedules the surrogate reproduces the simulator's ordering almost
exactly: Spearman ρ = 0.998, Kendall τ = 0.961, and the surrogate's top-1 choice is the
simulator's top-1 choice (Fig. 2a). The signal-to-error ratio — the spread of the objective across
the pool divided by the mean surrogate error — is 132.

We then took the ten distinct schedules produced by a multi-start gradient optimiser run **on the
surrogate itself**, all within 0.03 % of the baseline injection volume, and simulated each. In this
regime the ordering collapses: ρ = −0.21 (permutation p = 0.56, bootstrap 95 % CI
[−0.83, +0.65]), top-1 hit rate 0, and the surrogate's first choice is the simulator's *last*
(Fig. 2b). The mechanism is visible in the per-candidate optimism (Fig. 2c): the surrogate
overstates the objective by a median factor of 1.66 across the shortlist but by 10.6 for the
candidate it selects, and its predicted value correlates with its own error at r = 0.97. The
optimiser has concentrated the shortlist where the surrogate is optimistic — the optimiser's curse
— and selecting on the surrogate therefore selects for surrogate error. The resulting top-1 regret
is 1.39 % of the objective, or 46.0 M US$ at r = 8 %.

Two controls establish that the collapse is caused by the optimiser, not by top-*k* selection or
by the narrower objective spread. Ranking the surrogate's own top-10 out of the **random** pool
gives ρ = 0.95 with a top-1 hit rate of 1.0 over a comparable 5.0 % objective spread; ranking the
simulator's true top-10 out of the same pool gives ρ = 0.94.

Two design decisions follow, and both are enforced programmatically rather than by convention.
The surrogate is used **for screening and ranking only**; no absolute value it predicts is
reported anywhere in this paper. Every quantity we quote — including every ΔNPV in §4 — is read
from a full-physics run.

## 3.4 The Gaia decision framework

**Message protocol.** Every assertion produced inside the framework is a typed message carrying
`agent_id`, a stage label, a scalar confidence in [0,1] and a provenance record
(`source_id`, `source_region`, `source_type`, `source_sha256`). Messages failing validation are
rejected. The practical consequence is that **a role with nothing to cite cannot emit a legal
message**: whether a role is substantive is settled by the protocol rather than by argument.
`source_type` distinguishes deck, simulation, measurement, literature and reasoning, so a role
arguing from analogue literature is admissible but recorded as such.

**Cartographer.** Before any agent runs, an injector→producer influence matrix is derived from the
existing case library by partial regression of each producer's cumulative oil on each injector's
cumulative injection. The routine reports the design-matrix condition number and raises a
high-severity flag above 30, since inter-well attribution is untrustworthy under collinearity.

**Domain roles.** Seven roles each read a distinct slice: pressure and water-cut history; the
influence matrix together with fault transmissibility multipliers; per-well production
surveillance; the economic sweep; geomechanical bounds from analogue North Sea sandstone
literature; 4D time-lapse seismic priors; and realised-versus-target injection. Each role receives
a fixed prompt describing its data domain and a dynamic section showing **how the previously
adjudicated case behaved within that role's own domain**. Roles are given perspectives — what they
attend to and are accountable for — but deliberately **no directional stance**, so that
disagreement arises from differing evidence rather than from assigned bias. After an independent
pass, every role sees the others' assessments and must challenge the one it most disagrees with.

**Adjudication by evidence strength.** A synthesiser integrates the messages into candidate
schedules. It is instructed to weigh messages by `source_type` rather than by count: a concern
supported only by analogue literature does not override a positive finding supported by
simulation, and several roles voicing the same literature-based concern still constitute one piece
of analogue evidence. This rule was added after we observed that roles with no field-specific
measurement were flattening the main economic axis of the decision at equal weight with roles
holding measured data.

**Programmatic feasibility gate.** Candidates are screened by the surrogate under the economic
objective, then filtered by an arithmetic check — injection-budget deviation, box constraints on
θ, and saturation against measured per-well injectivity. This check is deliberately **not**
delegated to a language model: a language model can be argued with, an arithmetic bound cannot.

**Simulator adjudication and feedback.** Surviving candidates are run in OPM Flow. Vetoed
candidates are excluded from the run's score. The realised outcome becomes the next round's
dynamic input. The loop terminates when two consecutive rounds produce no high- or medium-severity
flag, rather than after a fixed number of rounds.

## 3.5 Evaluation protocol

**Arms.** Three arms are compared. The **agent team** runs all seven roles with cross-examination.
The **single agent** runs one role — the reservoir engineer — with the same shared background,
the same surrogate, the same feasibility gate and the same simulator adjudication. The **no-agent**
control is one-dimensional uniform scaling: a single multiplier applied to every injector in every
period, swept over seven values. It contains no language model and exposes exactly one decision
variable, which makes it the most restrictive comparator available; a high-dimensional control
method that cannot beat it has no claim to the additional dimensions.

**Endpoints.** Arms are compared by the ΔNPV of the best simulator-adjudicated schedule in the
run. Runs in which every candidate was vetoed are recorded as zero improvement — the operator
would execute the baseline — rather than discarded, since discarding them would silently remove
the worst outcomes.

**Repetition and randomisation.** Because arm outcomes are stochastic through the language model,
the resolvable difference is established before any ablation is interpreted: repeating an
identical configuration yields a standard deviation of 51.3 M US$ in this setting, so single-run
ablation tables are uninformative and can invert with additional repeats. Each agent arm is
repeated ten times, with arm order interleaved and randomised rather than blocked, so that
time-varying factors do not align with arm identity.

**Stated non-equivalences.** Two systematic differences between the agent arms are not equalised
and are stated rather than concealed. The language-model call budget differs by roughly a factor
of five (48.4 versus 9.0 calls per run; 193 k versus 21 k prompt characters), and the
cross-examination round exists only where more than one role is present, since a single agent
cannot debate itself. The multi-role arm is therefore a **composite treatment**, and any
difference between the agent arms must not be attributed to role count alone. The two arms do
receive the same surrogate, the same feasibility gate and the same simulator budget.
