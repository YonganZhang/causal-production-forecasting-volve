# 5. Discussion

The results carry one methodological finding and one engineering outcome. The methodological
finding is that predictive accuracy and decision reliability are distinct properties of a
surrogate, and the second does not follow from the first (§4.1). The engineering outcome is that a
multi-agent framework built around that distinction produced simulator-verified value of
+361.3 M US$ at 2.7 full-physics evaluations per run (§4.2, §4.3). We take these in turn, then the
question of whether the decisions are inspectable, then the boundary of the evidence.

## 5.1 Predictive accuracy is not decision reliability

The surrogate reproduces the simulator's ordering of randomly sampled schedules almost exactly
(ρ = 0.998, top-1 hit rate 1.0) and its point predictions to 0.369 %. Yet on the ten schedules
produced by optimising against that same surrogate, its ordering carries no usable signal
(ρ = −0.21) and its first choice is the simulator's last.

The mechanism is not extrapolation error in the ordinary sense. Point accuracy degrades by a factor
of four (0.369 % → 1.44 %), which alone would not overturn an ordering spread over 1.4 % of the
objective. What overturns it is that the error becomes *correlated with the decision*: predicted
value and own error correlate at r = 0.97 across the shortlist, and the candidate ranked first is
the one overestimated most, by 10.6 against a shortlist median of 1.66. An optimiser searching
against a surrogate does not sample its error distribution at random; it seeks the region where the
surrogate is optimistic. Selecting on the surrogate therefore selects for surrogate error — the
optimiser's curse, measured here directly against a full-physics arbiter rather than assumed.

Two controls rule out the obvious alternative, that any restriction to top candidates compresses
rank correlation. Drawing the surrogate's own top-10 from the *random* pool preserves ρ = 0.95 with
a top-1 hit rate of 1.0 across a comparable objective spread, and drawing the simulator's true
top-10 from the same pool gives ρ = 0.94. The collapse appears only under optimisation.

The practical implication is a validation gap rather than a modelling defect. A surrogate validated
on independently sampled data has been shown to predict well; it has not been shown to *choose*
well among the candidates an optimiser will actually produce. These are different tests, and only
the second is relevant to a decision pipeline. We do not claim this as a general property of
surrogate-assisted reservoir optimisation — the evidence is one field, one surrogate architecture,
one optimisation set-up, and an optimisation-region sample of ten with a wide confidence interval.
What it does support is that high global accuracy did not guarantee reliable decision ranking in
the optimiser-generated region in this case, which is sufficient to make adjudication a design
requirement.

## 5.2 What a layered agent team contributes

Given §5.1, the architecture's organising question is not *how capable is each component* but
*what is each component competent to decide*. Gaia answers it by layer. The data agents produce
and characterise evidence, and the cartographer reports the condition number that says when its own
output should not be trusted. The knowledge agent supplies each reasoning role with what it may use
and, equally, with what it must not claim — curating absence is what makes evidence-weighted
adjudication possible at all, because a role that declares it holds no field measurement can be
discounted, whereas a role that quietly invents numbers cannot. The reasoning roles argue from
disjoint evidence under a protocol in which an uncited assertion is not a legal message. The
evaluation layer screens at 18.1 µs and gates on arithmetic. The simulator prices.

That this allocation is load-bearing rather than diagrammatic is visible in the loop's behaviour.
All ten runs ended at or above their starting point (§4.3, Fig. 5c). An earlier configuration — identical
surrogate, gate and simulator — degraded across rounds in six of seven runs, a net −139.6 M US$,
because roles holding no field-specific measurement were weighted equally with roles holding
simulated evidence. Changing how adjudication treats evidence changed the sign of the outcome (Fig. 3c).

Two cautions on strength. The progression is consistent end-to-end but not monotone. And the
before/after comparison of adjudication rules is not a randomised contrast within a single batch,
so it indicates a mechanism rather than estimating an effect size.

Reservoir engineering has adopted machine learning widely for prediction, history matching and
proxy-assisted optimisation, but agent-based decision systems remain uncommon in the field. The
contribution we would emphasise is therefore not that language models can reason about reservoirs,
but that a division of authority grounded in measured competence — with the physical model
retaining the right to assign value — produced verified economic gains in a real field model.

## 5.3 Are the decisions inspectable?

A 24-dimensional schedule chosen by a multi-agent system is of limited use to an operator who
cannot see what it is doing. The front-loading analysis of §4.4 addresses this: on held-out random
cases the scalar τ correlates with ΔNPV at ρ = +0.714, and the accepted schedules sit on the
positive side of that axis.

The direction itself is unsurprising — injecting earlier serves both sweep efficiency and
discounting, and a reservoir engineer would expect it. We regard that as a strength. It means a
high-dimensional policy produced by agent reasoning compresses, in substantial part, onto a
direction that is legible in reservoir terms: the framework did not earn its gain through an opaque
exploit that an operator would be unable to sanction. The finding is the presence of interpretable
economic structure inside an AI-generated high-dimensional decision, not a new reservoir law. It
remains an association; τ was not intervened upon and no cross-field evidence is offered.

## 5.4 Boundary of the evidence

**Reference strategy.** One-dimensional uniform scaling is an intentionally constrained reference,
not a state-of-the-art optimiser benchmark. It quantifies the gain from moving to structured
high-dimensional decision making. Two factors change together across that comparison — the presence
of the agent team and the expansion from one to twenty-four decision variables — and the experiment
does not separate them. We therefore do not claim superiority over conventional reservoir
optimisation methods; no such optimiser was benchmarked here.

**Single field, single configuration.** All results come from the Norne model under one economic
setting, one surrogate architecture and one optimisation set-up. The ranking collapse of §5.1 in
particular has not been reproduced across surrogates or fields.

**No human baseline.** The framework is compared with an algorithmic reference only; nothing here
speaks to how it compares with an experienced reservoir engineer.

**Offline knowledge curation.** The knowledge layer runs once, before the decision loop, and its
outputs are fixed across runs. A framework that re-curates knowledge inside the loop is a different
system and is not evaluated here.

**Interpretation, not identification.** τ is an association (§5.3), and the influence matrix of
§4.5 is a partial regression on simulated cases rather than a tracer measurement.

**Historical field.** Norne's production history has already occurred. The economic figures are
counterfactual evaluations within a validated full-physics model, not field-validated outcomes.
The contribution offered here is the framework and its evidence, with the field as test bed.
