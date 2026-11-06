# P1-S1 — continuous sensing decisions from frozen shared fields

Every automatic tick counts toward utility, including uncommitted predictions. A shared0.1 bootstrap sensor cost per episode is additional and equal across strategies. Actions select recorded sensors; they do not control physical motion. All world weights remain fixed and calibration people are separate from training, selection and final test people.

| World run | Learned total cost | Direct-sensor actor | Strongest fixed / cost | Paired cost gain95% interval | Commitment coverage | Gate |
|---|---:|---:|---|---|---:|---|
| EXP-T2-801 | 0.4411 | 0.4513 | listen / 0.4551 | +0.0140 [-0.0103,+0.0324] | 60.6% | FAIL |
| EXP-T2-802 | 0.4424 | 0.4487 | listen / 0.4556 | +0.0133 [-0.0014,+0.0248] | 58.3% | FAIL |
| EXP-T2-803 | 0.4471 | 0.4375 | listen / 0.4495 | +0.0023 [-0.0220,+0.0204] | 65.2% | FAIL |
| EXP-T2-804 | 0.4635 | 0.4495 | listen / 0.4630 | -0.0005 [-0.0336,+0.0268] | 55.6% | FAIL |

The strongest fixed baseline is the lowest cost across all registered fixed strategies. The actor receives current processed-field/memory means, model risk and previous sensor bits; future labels enter only train-only risk fitting and final scoring. Standardized128-hidden heads predict counterfactual future error after a2-packet guard. Goals are explicit audio/video/both enums, not learned language instructions. Summary JSON additionally reports active/quiet tick-conditional means and risk calibration error. Own-speaker activity is an evaluation-only label; primary utility retains every tick.

The12 fresh people belong to only three independent meetings. Registered gates use person-cluster intervals; summary.json also gives pre-test-specified meeting-cluster sensitivity intervals, including the field-versus-raw comparison. Three meetings provide limited evidence about new environments.

A physical-action world model still needs action-conditioned physical transitions, goals/reward, multi-step planning and embodiment-specific data. This recorded-data sensing loop cannot establish those capabilities. See the paced CPU audit for automatic action ticks, deadlines and exact weight invariance.
