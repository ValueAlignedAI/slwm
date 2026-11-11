# Matched context training changes the architecture conclusion

Each pair saw identical observations, previous actions, commands, labels, masks and 41,943,040 clock positions. G10 changed the model's effective training history by segmenting those same continuous streams at 32 steps, cycling through 32/64/128/256/512, or leaving all 512 steps intact. All 27 13,245,696-parameter models trained from scratch. Each fit saw 61,440 independently initialized simulator worlds and 20,480 sampled long room windows. A complete continuous training stream has fewer independent initializations per sampled position than G9's short independent episodes.

## Uninterrupted decisions

The metric is percent of time at the requested station, with full causal attention available and recurrent state carried for 512 decisions. Seed order is 1001, 1002 and 1003. Each training seed faces the same 256 held missions, with variable goal intervals and unseen intermittent stalls. The controller that decodes the sensors directly achieved 95.64%. Always holding achieved 5.10%.

| Architecture | Train 32 | Train variable | Train 512 |
| --- | ---: | ---: | ---: |
| SLWM | 95.3 / 95.4 / 94.4 | 93.7 / 94.0 / 93.4 | 92.7 / 93.2 / 92.4 |
| Dense GPT-2 | 54.7 / 66.2 / 67.5 | 90.8 / 94.2 / 94.9 | 73.6 / 51.1 / 58.4 |
| Gated recurrent | 94.3 / 93.9 / 94.2 | 94.3 / 94.0 / 94.0 | 94.3 / 93.5 / 94.0 |

Under variable-context training, the registered SLWM-minus-dense gaps were +2.85 [1.95, 3.80], −0.21 [−0.63, 0.25], and −1.50 [−1.89, −1.15] percentage points. The primary criterion required positive paired intervals in all three seeds and passed in one. Dense GPT-2 benefited more from seeing several history lengths than from training exclusively at the longest length. The gated reference was comparatively insensitive to segmentation. We bootstrap whole missions 4,000 times within each training seed. [Scores and paired intervals](metrics/context.json) · [Mission-level arrays](metrics/context-paired.npz)

## Recalibration and memory

On a separate 256-mission cohort with two actuator changes, the variable-context hybrid reached 90.7 / 91.0 / 90.6%, dense GPT-2 reached 85.1 / 87.7 / 90.6%, and the gated recurrent control reached 94.2 / 93.7 / 94.0%. The physical reference reached 95.49%. A neutral device-change notice exposed the need to reassess the equipment without specifying the new signs.

Good interactive control did not guarantee correct recovery of the original tap in a stationary, sensor-identical history. At a 480-tick delay, the variable-context hybrid selected the right action in 55.1 / 57.0 / 37.1% of cases. Dense GPT-2 reached 64.8 / 48.8 / 48.4%, and the gated reference reached 50.8 / 51.2 / 52.3%. Only the sign of an earlier issued control distinguishes each pair. A policy given identical present evidence can obtain at most 50% by choosing the same action in both worlds. Live movement provides further evidence that these diagnostic pairs deliberately remove. [Delayed and wrong-history controls](metrics/context.json)

G9 showed an advantage for its hybrid under exclusively short training and uninterrupted deployment. G10 reproduces the large gap under short training, then shows that it does not persist across seeds after matching variable-context training. G10 also uses new trajectories, so changes across study generations should not be read as an isolated intervention. The within-G10 factorial isolates the segmentation decision. Independent real-recording alignment and transient-instruction tests address separate claims. [Verified weight and optimizer identities](metrics/context-artifacts.json) · [Transient instruction registration](instruction-memory.md)
