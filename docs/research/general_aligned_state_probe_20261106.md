# Can the aligned space carry decisions?

Prospective supplemental probe, registered after G1 core fitting began but
before any real readout fit, calibration prediction, final embedding or final
score. It changes no core weight, input data, primary case, baseline choice or
primary success criterion. The original512D last-state probe remains primary.

Sharing processor weights does not guarantee one sufficient world state. G1's
128D normalized alignment projection is trained on independently encoded paired
modalities; dense prediction heads read the larger field. A good retrieval
projection might discard pose, timing or action-relevant information. Conversely,
the larger field could retain private modality details while the alignment head
alone creates a common space.

Add the frozen128D alignment embedding as a secondary feature condition, using
exactly the same registered examples, training-only projection,128-hidden readout,
two seeds, selection roles and target scaling as the original last-state probe.
No future query/action plan enters either actor feature. Cache both in the same
forward pass. Keep specialist, raw-history and edge controls. Do not relabel a
secondary success as passage of the primary gate.

Interpret outcomes jointly with source-hidden transfer, sound→image semantic
retrieval, action-consequence prediction and the no-future/alignment controls.
If retrieval works while aligned-state decisions fail, the shared semantic space
is not yet a sufficient control state. If only large private states help, test
predictive training through an explicitly common bottleneck next. If both work,
test longer memory and actual action-dependent feedback before a robotics claim.

Related evidence motivates these distinctions: [ImageBind](https://arxiv.org/abs/2305.05665)
shows modality binding through naturally paired observations, with strong large
vision-language encoders. [V-JEPA2](https://arxiv.org/abs/2506.09985) combines
observation pretraining with action-conditioned post-training for robotic
planning. [DreamerV3](https://arxiv.org/abs/2301.04104) learns behavior using imagined
environment trajectories. Our small, from-scratch, partly unpaired study does
not reproduce their scale or establish closed-loop control. The proposed probe
asks whether its shared representation contains the information a decision
readout actually needs.
