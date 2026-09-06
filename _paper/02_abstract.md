# Abstract

Reservoir decisions require many expensive full-physics evaluations, and learned surrogates are the
standard means of affording them. Whether a surrogate validated for prediction can be trusted to
*choose* among optimiser-generated candidates has received little scrutiny. On the Norne field
model we show that it cannot be assumed: an operator surrogate that reproduced the simulator's
ranking of 500 independently sampled injection schedules at Spearman ρ = 0.998, with a top-1 hit
rate of 1.0 and a field-level error of 0.369 %, ranked the ten schedules produced by optimising
against itself at ρ = −0.21 with a top-1 hit rate of 0, selecting the candidate whose objective it
overstated most — by a factor of 10.6 against a shortlist median of 1.66; two controls place the
cause in the optimiser rather than in top-*k* selection. We therefore propose **Gaia**, a layered
agent team in which decision authority follows demonstrated competence. Fourteen agents across five
layers convert the raw simulation deck into a structured case library and an inter-well influence
map, curate per-role domain knowledge together with explicit statements of what each role does
*not* have evidence for, reason from disjoint evidence under a provenance-bearing message protocol,
screen candidates at 18.1 µs each and gate them on arithmetic feasibility — while the full-physics
simulator alone assigns value. Applied to water-injection scheduling as a case study, Gaia produced
simulator-verified improvements of 361.3 M US$ in NPV at 8 % over the historical schedule, against
62.6 M US$ for one-dimensional uniform scaling, using 2.7 full-physics evaluations per run against
7 for the reference sweep; all ten runs finished at or above their starting point. The best
schedule raises cumulative oil by 10.9 % while injecting 6.5 % less water, and remains profitable
at discount rates from 0 % to 15 %. The accepted
schedules concentrate on one interpretable direction, earlier rather than later injection
(ρ = +0.714 on held-out cases), so the recommendations remain legible to an engineer. The framework
is released as a reference implementation with each layer exposed as a callable agent.

**Keywords:** multi-agent framework; reservoir decision making; surrogate model; large language
model agents; waterflood optimisation; Norne field
