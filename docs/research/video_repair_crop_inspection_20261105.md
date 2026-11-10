# T2-V1 recorded-crop inspection — post-hoc limitation

Inspection occurs after weights, final feature scores and speech diagnostics are
fixed. No test crop, model, confidence threshold or gate changes. All 32 recorded
mouth strips remain available in the fixed listening gallery; these illustrations
are not a newly selected performance benchmark.

The fixed 96×64 crop contains visible speaking motion, but mouth position and size
vary substantially. In the CREMA example, lips sit close to the top edge. In the
RAV examples, the vertical position and scale differ. In particular, the saved
`occluded` diagnostic replaces only columns 16–79 and rows 12–39. It therefore
does **not** remove all lip pixels in every crop. Interpret it as a fixed central
rectangle intervention, not a reliable full-mouth ablation. The main success
gate uses additional-audio, wrong-utterance, real shifts and static controls;
it does not depend on this occlusion result.

The CREMA illustration is the first fixed gallery case:

![Recorded CREMA frames](../../experiments/multimodal/t2/video-repair-20261105/exploration/1026_DFA_FEA_XX/recorded_mouth.png)

These RAV illustrations include a positive person 09 and the two people 10/21
whose average effects are negative in both seeds. They were chosen post-hoc
to inspect that failure pattern. Lips remain visible in the failed examples,
so these observations do not establish cropping as the cause of the failures.

![Recorded RAV09 frames](../../experiments/multimodal/t2/video-repair-20261105/exploration/01-01-01-01-01-01-09/recorded_mouth.png)

![Recorded RAV10 frames](../../experiments/multimodal/t2/video-repair-20261105/exploration/01-01-01-01-01-01-10/recorded_mouth.png)

![Recorded RAV21 frames](../../experiments/multimodal/t2/video-repair-20261105/exploration/01-01-01-01-01-01-21/recorded_mouth.png)

A future geometry/occlusion study should specify its crop using training and
validation people, verify lip coverage before fitting, account for any added
face encoder, and reserve a new final holdout. The current result establishes
a contribution from synchronized recorded video on reused CREMA; it does not
isolate lip shape from other correlated visible motion.
