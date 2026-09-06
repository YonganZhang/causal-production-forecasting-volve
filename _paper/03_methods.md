# 3. The Gaia Agent Team

## 3.1 Overview

Gaia is a layered multi-agent framework for reservoir decision making. Agents span the whole
chain — raw simulation deck to structured case library, curated domain knowledge, multi-role
reasoning, fast evaluation and physical adjudication — rather than occupying only the reasoning
step (Fig. 1, Table 1). Fourteen agents are organised into five layers, and the design principle
running through all of them is that **authority is allocated according to demonstrated
competence**, not according to how capable a component appears:

- the **surrogate** has speed but no pricing authority, because §4.1 shows its ranking fails in
  precisely the region an optimiser explores;
- the **language agents** have reasoning but no veto, because feasibility is decided by an
  arithmetic gate that cannot be argued with;
- the **full-physics simulator** has pricing authority, and is the sole source of every economic
  number reported in this paper.

Water-injection scheduling on the Norne field is the case study through which the framework is
evaluated. It is the test bed, not the object of the design.

## 3.2 Problem setting and economic objective

The decision variable is a piecewise-constant injection schedule

  θ ∈ R^(S×I),  S = 6 control periods (2007, 2008, 2010, 2012, 2014, 2017),  I = 4 injectors,

where θ_{s,i} is the base-10 multiplier applied to injector *i*'s baseline rate during period *s*
over a 13-year forecast horizon on the Norne model (46 × 112 × 22 grid, 44,431 active cells, four
injectors F-1H–F-4H, 22 producers). History is simulated once; every candidate branches from the
same restart state.

The objective is net present value under an explicit water cost,

  NPV(r) = Σ_t [ q_o(t)·p_oil(t) − q_wi(t)·c_inj − q_wp(t)·c_prod ] · (1+r)^(−t),

with p_oil from EIA annual Brent spot prices, and c_inj, c_prod the per-barrel costs of injected
and produced water. We report r = 8 %, c_inj = 2.0 US$/bbl, c_prod = 1.0 US$/bbl. The historical
schedule gives NPV_8% = 1,773.4 M US$, the reference for every ΔNPV below.

Pricing water is not cosmetic. Under oil revenue alone, *how much water to inject* degenerates —
water is free and monotonically increases oil, so the optimum sits on the boundary and a fixed
injection budget must be imposed as a hard constraint standing in for a missing cost term.
Charging for water makes total injection an endogenous decision.

All cumulative quantities are read from the simulator's own running totals (FOPT, FWIT, FWPT)
rather than integrated from well curves. The two conventions disagree by 0.58 % on average, and
in 14 cases returned opposite feasibility verdicts for the same schedule.

## 3.3 Layer 1 — Data agents

**Data engineer.** Converts the raw deck and simulator output into a structured case library:
7,400 training samples and 500 confirmation samples generated independently *after* model
selection was frozen, each pairing a control vector with multi-well, multi-phase trajectories.

**Cartographer.** Recovers injector→producer influence from the case library by partial regression
of each producer's cumulative oil on each injector's cumulative injection, and reports its own
reliability: the design-matrix condition number, with a high-severity flag above 30. On 471 cases
the condition number is 2.05, so inter-well attribution is admissible here. This is an agent that
states when it should not be trusted.

## 3.4 Layer 2 — Knowledge agent

Domain roles cannot reason from a simulation deck directly; they need to know what the field is,
what data exist for it, and — critically — what data do *not*. A knowledge-curation agent (an LLM
agent, Claude Opus 5) performs three tasks and emits one knowledge base per role:

1. **Deck audit.** Enumerates which physical descriptions the deck contains (PERMX, PORO, NTG,
   FAULTS/MULTFLT, well controls) and which it does not (GEOMECH, STRESS, YOUNGMOD, POISSON).
2. **Field literature retrieval.** Queries bibliographic services for field-specific work; for
   Norne this returns time-lapse (4D) seismic studies, supplied to the corresponding role as
   titles and DOIs only.
3. **Knowledge-base emission with explicit absences.** Each role receives what it may use *and* a
   statement of what it must not claim. The geomechanics role, for instance, is told that the deck
   carries no measured stress data for this field, that it may reason only by analogy to North Sea
   sandstone reservoirs, and that asserting measured Norne stress values is prohibited. Its
   provenance is recorded as `literature`, not `measurement`.

Curating absence is what makes the evidence-weighted adjudication of §3.5 possible: a role that
declares it has no field measurement can be weighted accordingly, whereas a role that quietly
invents numbers cannot.

This layer runs **once, offline**, and its outputs are consumed unchanged by every decision run
reported here. The side effect is useful: because the knowledge bases are fixed across
repetitions, run-to-run variation is not confounded by knowledge drift.

## 3.5 Layer 3 — Reasoning agents

**Message protocol.** Every assertion is a typed message carrying `agent_id`, a stage label, a
confidence in [0,1] and a provenance record (`source_id`, `source_region`, `source_type`,
`source_sha256`); messages failing validation are rejected. A role with nothing to cite cannot
emit a legal message, so whether a role is substantive is settled by the protocol rather than by
argument. `source_type` distinguishes deck, simulation, measurement, literature and reasoning.

**Seven domain roles.** Reservoir engineering (pressure and water-cut history); connectivity (the
influence matrix and fault transmissibility multipliers); production surveillance (per-well rates
and bottom-hole pressures); economics (price deck, discount rates, water costs); geomechanics
(analogue literature bounds); 4D seismic (time-lapse priors); and constraint auditing (realised
versus target injection) as an internal red team. Each role reads only its own knowledge base plus
how the previously adjudicated case behaved *within its own domain*. Roles are given perspectives
but deliberately **no directional stance**, so disagreement arises from differing evidence rather
than assigned bias. After an independent pass, every role sees the others' assessments and must
challenge the one it most disagrees with.

**Synthesiser.** Integrates the messages into candidate schedules, weighting by `source_type`
rather than by count: a concern supported only by analogue literature does not override a positive
finding supported by simulation, and several roles voicing the same literature-based concern still
constitute one piece of analogue evidence. This rule was introduced after roles holding no
field-specific measurement were observed flattening the main economic axis of the decision at
equal weight with roles holding simulated evidence.

## 3.6 Layer 4 — Evaluation agents

**Surrogate.** A full-physics run takes approximately one minute, so bulk screening requires a
stand-in. We train a parametric operator mapping the static control vector directly to
trajectories,

  f_φ : R^24 → R^(W×P×T),  W active producers, P ∈ {oil, water}, T = 40 time nodes.

This is operator learning, not time-series forecasting: no historical sequence is supplied at
inference. The architecture is a conditioning MLP followed by a Transformer encoder over time
nodes with Fourier positional embeddings (16 bands). On the held-out confirmation set the
field-level relative error is 0.369 % and R² = 0.9959, at 18.1 µs per candidate — about
3.3 × 10⁶ times faster than the corresponding simulation.

**The surrogate is used for screening and ranking only**, and no absolute value it predicts is
reported anywhere in this paper. §4.1 shows why this restriction is a requirement rather than
caution.

**Feasibility auditor.** Surviving candidates pass an arithmetic gate: injection-budget deviation,
box constraints on θ, and saturation against measured per-well injectivity. This check is
deliberately not delegated to a language model — a language model can be argued with, an
arithmetic bound cannot.

## 3.7 Layer 5 — Adjudication

Candidates clearing the gate are run in OPM Flow. Vetoed candidates are excluded from the run's
score; the realised outcome becomes the next round's input to every role. The loop terminates when
two consecutive rounds produce no high- or medium-severity flag, rather than after a fixed number
of rounds. The simulator is not an agent: it is the arbiter, and every economic figure in §4 is
one of its outputs.

## 3.8 Evaluation protocol

**Reference strategy.** The framework is compared against one-dimensional uniform scaling: a
single multiplier applied to every injector in every period, swept over seven values. It contains
no learned or language model and exposes exactly one decision variable, which makes it the most
restrictive comparator available; a high-dimensional control method that cannot beat it has no
claim to the additional dimensions.

**Endpoints and repetition.** Runs are scored by the ΔNPV of the best simulator-adjudicated
schedule. Runs in which every candidate was vetoed count as zero improvement — the operator would
execute the baseline — rather than being discarded. Because the language agents are stochastic, the
resolvable difference is established before interpreting any comparison: repeating an identical
configuration yields a standard deviation of 51.3 M US$ in this setting. The framework is run ten
times, in interleaved randomised order rather than in blocks, so that time-varying factors do not
align with configuration.

**Software and models.** Simulation uses OPM Flow 2026.04. All language agents run on Claude
Opus 5. A reference implementation of the framework, with each layer exposed as a callable agent,
is released as the `gaia_sdk` package; its knowledge layer reproduces byte-for-byte the knowledge
bases consumed by the runs reported here.
