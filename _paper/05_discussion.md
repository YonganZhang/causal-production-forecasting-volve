# 5. Discussion

## 5.1 What the surrogate is for, and what it is not for

The surrogate reaches 0.369 % field-level error and runs 3.3 × 10⁶ times faster than the
simulator, yet we never report a number it produced. This is not excessive caution. Accuracy
measured on randomly drawn schedules does not transfer to the region an optimiser drives towards:
in a separate experiment the same model overstated the objective gap at optimiser-located extrema
by a factor of 4.1 and ranked the truly best candidate fourth of four. A framework that trusted
surrogate rankings would be misled exactly where the decision is made.

The resolution adopted here is a division of labour rather than a better surrogate. The surrogate
absorbs the cost of breadth — thousands of candidate evaluations at microsecond cost — and the
simulator retains sole authority over value. This assigns each component the regime in which it is
reliable, and it is the reason the loop can afford to be exploratory without becoming
untrustworthy.

## 5.2 Structure buys reliability, not return

The clearest quantitative result is also the narrowest: role structure reduces dispersion
(Levene p = 0.001 and p = 0.002 against the unstructured arm) without a resolvable change in mean
outcome, and the number of roles does not matter (p = 0.667 between the seven-role and single-agent
arms). The unstructured arm is not worse on average; it is less predictable, and its failures are
qualitatively different — several runs end at or near zero improvement because every proposal is
rejected by the feasibility check.

For field decision support this asymmetry matters more than the mean. An operator does not sample
repeatedly from a decision procedure and keep the best draw; a single plan is executed. A method
whose realised value ranges over a factor of several, with occasional total failures, is difficult
to adopt regardless of its expected value.

The cost side must be stated with equal clarity. The seven-role arm consumes 48.3 language-model
calls per run against 6.4 for the unstructured arm — a 7.5-fold cost purchasing reduced dispersion
and no measured gain in return. Whether that trade is worthwhile depends on the consequences of a
bad plan in the deployment setting, not on the numbers reported here.

Three differences between arms were not equalised: call budget, the presence of a
cross-examination round, and the number of roles. The multi-role arm is therefore a composite
treatment. We regard the cross-examination round as intrinsic to having a team — a single agent
cannot debate itself — but the call-budget difference remains a genuine confound, and the
possibility that dispersion is reduced by additional deliberation rather than by role structure
per se cannot be excluded by this design.

## 5.3 Effective control freedom is much smaller than nominal control freedom

Target injection rates span a factor of 430 across the explored schedules, while realised rates
never exceed roughly 20,000 Sm³/day: every injector saturates against its measured injectivity.
F-4H is the extreme case, pinned near 593 Sm³/day and carrying negative influence coefficients for
three producers, so raising its target neither increases injection nor increases oil.

This has a direct methodological consequence. Optimisation performed in target space silently
allocates most of its search to directions that cannot be realised, and any attribution computed
in that space will misstate which wells matter. Reporting realised rather than target quantities
is therefore not a reporting preference but a correctness requirement — a point reinforced by the
0.58 % disagreement we measured between simulator-reported cumulative volumes and volumes
integrated from well curves, which is one to two orders of magnitude above the tolerances used
elsewhere in this work.

## 5.4 Water must be priced for the injection question to be well posed

With oil revenue alone, total injection has no interior optimum: water is free and monotonically
increases oil, so the optimiser drives to the boundary. Earlier formulations therefore imposed a
fixed injection budget — a hard constraint standing in for a missing cost term. Once water carries
a price, the optimal total injection moves from +5.4 % of baseline at zero water cost to −30.6 % at
4.0/3.0 USD/bbl. The reversal is not a sensitivity artefact; it is the question becoming
well posed.

## 5.5 Ablation of language-model systems requires a repeatability floor

A single-run-per-configuration ablation of the seven roles indicated that every role contributed
positively, with marginal values from +10 to +148 M$. Repeating an identical configuration
produced a standard deviation of 35.9 M$ over five runs and 51.3 M$ pooled over six per arm; every
apparent contribution then fell inside the noise, and the role that had appeared strongest
(+102.6 M$ at n = 1) measured +15.6 M$ at n = 6.

We could find no reported practice of establishing this floor before interpreting ablations of
multi-agent systems, yet without it an ablation table can inverted by additional repeats. At the
dispersion observed here, resolving 30 M$ differences would require roughly 24 repeats per arm.

A related failure was diagnostic rather than statistical. In an intermediate version, the roles'
data slices were not updated between rounds: each analyst re-read the same baseline case every
round, so the same conclusions were restated verbatim across rounds while only the synthesizer saw
new evidence. The orchestration diagram was correct — proposals, adjudication and feedback were all
present — but feedback reached only one participant. Diagrams do not establish that a loop is
closed for every participant; per-role inputs must be inspected.

## 5.6 Limitations

Results are specific to this reservoir model and well configuration and are not extrapolated to
other fields. The counterfactual cannot be validated in the field: Norne is decommissioned, so the
model is a testbed rather than a recipient of recommendations. Oil and water prices enter only the
evaluation, never the search, which keeps the reported reversals properties of the physics rather
than of a fitted price path. Facility limits on injection rate beyond measured injectivity are not
modelled. The geomechanical role reasons from analogue North Sea sandstone literature because the
model carries no stress or elastic-property data; its statements are recorded as literature-derived
and are qualitative bounds only. Finally, arm comparisons rest on twelve repeats each, sufficient
to separate dispersion but not means at the effect sizes observed.
