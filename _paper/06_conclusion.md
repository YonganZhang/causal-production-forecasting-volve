# 6. Conclusion

Reservoir decisions require many expensive full-physics evaluations, and learned surrogates are the
standard means of affording them. We asked what a surrogate, and an AI system built around one, may
and may not be trusted to decide, and built a framework that follows from the answer.

A surrogate qualified by conventional validation can screen but cannot decide. On 500 independently
sampled schedules our operator surrogate reproduced the simulator's ranking at Spearman ρ = 0.998
with a top-1 hit rate of 1.0 and a field-level error of 0.369 %, at 18.1 µs per candidate against
approximately one minute per simulation. On the ten schedules produced by optimising against that
same surrogate, its ranking carried no usable signal (ρ = −0.21), its first choice was the
simulator's last, and the candidate it selected was the one it overestimated most — by a factor of
10.6 against a shortlist median of 1.66, with predicted value and own error correlated at r = 0.97.
Two controls placed the cause in the optimiser rather than in top-*k* selection.

We therefore propose **Gaia**, a layered agent team in which authority follows demonstrated
competence. Fourteen agents across five layers turn the raw deck into a structured case library and
an influence map, curate per-role knowledge together with explicit statements of what each role
does *not* have evidence for, reason from disjoint evidence under a provenance-bearing protocol,
screen candidates at 18.1 µs and gate them on arithmetic — while the full-physics simulator alone
assigns value.

Applied to water-injection scheduling on the Norne field, the framework produced simulator-verified
improvements of +361.3 M US$ in NPV at 8 % over the historical schedule, against +62.6 M US$ for
one-dimensional uniform scaling, using 2.7 full-physics evaluations per run against 7 for the
reference sweep. All ten runs finished at or above their starting point, and the accepted schedules
concentrate on one interpretable direction — earlier rather than later injection (ρ = +0.714 on
held-out random cases) — so the recommendations are legible to an engineer rather than opaque.

The practical implication is a division of authority rather than a division of labour: learned
models and language agents may generate, curate and narrow, and the physical model retains the
right to assign value. Conventional surrogate validation, performed on independently sampled data,
does not establish that a surrogate is safe to decide with, because it does not test it on the
candidates an optimiser will actually produce.

These results are bounded. The reference strategy is intentionally constrained rather than a
state-of-the-art optimiser, and the comparison does not separate agent reasoning from the enlarged
decision space. All evidence comes from one field, one surrogate architecture and one economic
setting; the knowledge layer runs offline rather than inside the loop; and Norne's production
history has already occurred, so the economic figures are counterfactual evaluations inside a
validated model rather than realised outcomes.

Three directions follow. Testing whether the screening–decision gap reproduces across surrogate
architectures and fields would establish whether it is a property of this model or of
surrogate-assisted optimisation generally. Benchmarking the loop against conventional
derivative-free optimisers under a matched full-physics budget would convert the efficiency
observation into a claim about sample efficiency. And moving knowledge curation inside the loop —
so that agents refresh field knowledge as evidence accumulates — is, we think, a more promising
direction than adding further reasoning roles.
