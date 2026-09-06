# 6. Conclusion

High-dimensional, multi-period waterflood scheduling requires many expensive full-physics
evaluations, and learned surrogates are the standard route to affording them. We asked what a
surrogate may and may not be trusted to do inside such a decision loop, and built the framework
that the answer implies.

The answer is that a surrogate qualified by conventional validation can screen but cannot decide.
On 500 independently sampled schedules our operator surrogate reproduced the simulator's ranking
at Spearman ρ = 0.998 with a top-1 hit rate of 1.0 and a field-level error of 0.369 %, at
18.1 µs per candidate against approximately one minute per simulation. On the ten schedules
produced by optimising against that same surrogate, its ranking carried no usable signal
(ρ = −0.21), its first choice was the simulator's last, and the candidate it selected was the one
it overestimated most — by a factor of 10.6 against a shortlist median of 1.66, with predicted
value and own error correlated at r = 0.97. Two controls placed the cause in the optimiser rather
than in top-*k* selection.

Building on that boundary, we organised surrogate screening, multi-role agent proposal, an
arithmetic feasibility gate and full-physics adjudication into a closed loop in which the
simulator alone assigns value. On the Norne field model the loop produced simulator-verified
improvements of +361.3 M US$ (seven-role team) and +347.4 M US$ (single role) in NPV at 8 %
against the historical schedule, compared with +62.6 M US$ for one-dimensional uniform scaling,
using 2.7 and 2.6 full-physics evaluations per run against 7 for the reference sweep. All twenty
runs finished at or above their starting point. Decomposing the reasoning into seven specialised
roles produced no detectable gain in mean performance over a single role (+13.8 M US$,
p = 0.63), which locates the source of the improvement in the closed loop itself rather than in
the number of roles. The accepted schedules concentrate on one interpretable direction — earlier
rather than later injection (ρ = +0.714 on held-out random cases) — so the gain is legible to an
operator rather than opaque.

The practical implication is a division of authority rather than a division of labour: learned
models and language agents may generate and narrow the candidate set, and the physical model
retains the right to assign value. Conventional surrogate validation, performed on independently
sampled data, does not establish that a surrogate is safe to decide with, because it does not test
it on the candidates an optimiser will actually produce.

These results are bounded. The reference strategy is intentionally constrained rather than a
state-of-the-art optimiser, and the comparison against it does not separate agent reasoning from
the enlarged decision space. Ten repetitions per arm cannot establish equivalence between the
agent arms. All evidence comes from one field, one surrogate architecture and one economic
setting, and Norne's production history has already occurred, so the economic figures are
counterfactual evaluations inside a validated model rather than realised outcomes. No human expert
baseline was run.

Three directions follow directly. First, testing whether the screening–decision gap of §4.1
reproduces across surrogate architectures and fields would establish whether it is a property of
this model or of surrogate-assisted optimisation generally. Second, benchmarking the loop against
conventional derivative-free optimisers under a matched full-physics budget would convert the
efficiency observation into a claim about sample efficiency. Third, the adjudication rule that
weighs assertions by evidence type changed the loop's direction in this study; making that rule
explicit and testable is, we think, more promising than adding further reasoning roles.
