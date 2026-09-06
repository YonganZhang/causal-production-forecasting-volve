# 1. Introduction

Deciding how to operate a waterflood — how much water to inject, into which wells, and when — is
among the recurring decisions in reservoir management, and one of the few whose economic
consequences are large enough to be worth optimising deliberately. Formulated over multiple wells
and multiple control periods, the decision space grows multiplicatively, and each candidate must be
evaluated by a full-physics reservoir simulation. On the model used here a single run takes
approximately one minute, so any method that requires substantial trial and error is impractical at
the simulator's own cost. This tension — a decision worth optimising, evaluated by a model too
expensive to query freely — motivates most of the machine-learning work now entering reservoir
engineering.

The standard response is a learned surrogate. A model trained on simulated cases replaces the
simulator during search, and is qualified in the conventional way: held-out cases are predicted,
and R², relative error or rank correlation are reported. If those metrics are good, the surrogate
is taken to be ready for optimisation.

That step contains an untested inference. A surrogate validated on independently sampled data has
been shown to *predict* well. It has not been shown to *choose* well among the candidates an
optimiser will actually generate — and those are not the same population. An optimiser does not
sample the surrogate's error distribution at random; it searches for the region the surrogate
believes is best, which is disproportionately the region where the surrogate is optimistic.
Selecting on the surrogate therefore selects, in part, for surrogate error. To our knowledge this
gap is rarely measured directly in reservoir applications, because what is usually reported is
predictive validation rather than decision validation.

We measure it. Using the same surrogate and the full-physics simulator as arbiter, we compare rank
fidelity on randomly sampled schedules with rank fidelity on the schedules an optimiser produces
when run against that surrogate. The result is not a gradual degradation but a reversal, and it
determines what follows: if the surrogate cannot be trusted to choose, the architecture must place
that authority elsewhere.

This reframes the design question. It is no longer *how capable can each component be made*, but
*what is each component competent to decide*. Learned models can evaluate at scale. Language agents
can integrate heterogeneous engineering evidence — pressure and water-cut history, inter-well
connectivity, production surveillance, price decks, analogue geomechanical bounds, time-lapse
seismic priors — and propose candidates that are structured rather than arbitrary. Neither can be
ground truth. Only the physical model can price a schedule.

We therefore propose **Gaia**, a layered agent team for reservoir decision making in which
authority follows demonstrated competence. Agents span the full decision chain rather than
occupying only the reasoning step: data agents build a structured case library and an inter-well
influence map and report their own reliability; a knowledge agent audits which physical
descriptions the simulation deck contains and which it lacks, retrieves field-specific literature,
and issues each reasoning role a knowledge base together with an explicit statement of what that
role has no evidence for; seven domain roles argue from disjoint evidence under a message protocol
in which an uncited assertion is not admissible; an evaluation layer screens candidates with the
surrogate and gates them on arithmetic feasibility; and the full-physics simulator adjudicates. The
right to assign value is never delegated to a learned or language model.

Agent-based systems remain uncommon in reservoir engineering, where machine learning has been
adopted mainly for prediction, history matching and proxy-assisted optimisation. Whether such
systems can create verified economic value in a real field model, rather than plausible-looking
recommendations, is an open question. We treat water-injection scheduling on the Norne field as the
case study for answering it. The field is the test bed, not the object of the design: its
production history has already occurred, and the economic figures reported here are counterfactual
evaluations within a validated full-physics model.

The contributions are:

1. **The Gaia layered agent architecture** — fourteen agents across five layers, covering data
   preparation, knowledge curation, multi-role reasoning, evaluation and adjudication, with
   decision authority allocated by measured competence rather than by apparent capability. A
   reference implementation is released with each layer exposed as a callable agent.

2. **A parametric operator surrogate** mapping static control vectors directly to multi-well,
   multi-phase production trajectories, with a field-level error of 0.369 % and R² = 0.9959 at
   18.1 µs per candidate — about 3.3 × 10⁶ times faster than the corresponding simulation.

3. **A measured capability boundary for that surrogate** — accurate enough to screen (ρ = 0.998 on
   random schedules) and not accurate enough to decide (ρ = −0.21 in the optimiser's own region,
   where it selects the candidate it overestimates most). This is the evidence on which the
   framework's allocation of authority rests.

4. **Simulator-verified economic value** — 361.3 M US$ in NPV at 8 % above the historical schedule
   at 2.7 full-physics evaluations per run, against 62.6 M US$ for one-dimensional uniform scaling
   at 7 evaluations, with all runs ending at or above where they started and the accepted schedules
   concentrating on one interpretable direction.

Section 3 describes the framework layer by layer, Section 4 reports the measured capability
boundary and the economic results, and Section 5 discusses what the layered allocation of authority
contributes and where the evidence stops.
