# 6. Conclusion

We presented a decision framework for waterflood scheduling in which a parametric operator
surrogate serves as a fast stand-in for the reservoir simulator, domain roles propose and
cross-examine schedules from separate data slices under a provenance-carrying message protocol, a
programmatic feasibility check rejects unexecutable proposals, and a full-physics simulator retains
sole authority over value. Realised outcomes return to each role within its own data domain,
closing the loop for every participant rather than for the synthesizer alone.

On the Norne model the surrogate attains 0.369 % field-level error at 18.1 µs per candidate,
approximately 3.3 × 10⁶ times faster than simulation, and its error against adjudicated outcomes
stays within ±1.3 % in the loop. Role structure reduces the dispersion of realised value
significantly (Levene p = 0.001 and p = 0.002 against an unstructured baseline) without a
resolvable change in mean, and the number of roles is immaterial (p = 0.667): the distinction that
matters is whether role structure is present, not how much of it there is. This reliability is
purchased at 7.5 times the language-model cost.

Two findings constrain how such systems should be evaluated. Effective control freedom is far
smaller than nominal control freedom — target injection rates span a factor of 430 while realised
rates saturate — so optimisation and attribution must be expressed in realised quantities. And
ablations of language-model systems are uninterpretable without first measuring repeatability: at
the dispersion observed here, single-run ablation tables invert under additional repeats.

The immediate open question is whether reduced dispersion is attributable to role structure itself
or to the additional deliberation that accompanies it; separating the two requires a budget-matched
control that this study does not provide.
