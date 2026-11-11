# G11 primary result · what follows a spoken instruction?

None of the 18 paired architecture–seed–fold cells met the registered next-verb criterion. The physical endpoint also passed in zero cells. The experiment compares the same sensor prefix with the actual recorded directive, a different earlier directive spoken in the same recording, and no directive. The interventions change the model's evidence. The underlying human trajectory was recorded once. These null results bound what the tested predictors can use.

## Next action

Fold A held out both coffee machines and computers. Fold B held out printers and cameras. None of the 96 primary recordings appeared in G10. Seeds 1001, 1002 and 1003 share the published G10 variable-context parents across matched treatments. The held sets contain 403 and 435 qualified prospective decisions. The main accuracy is calculated across recordings within each family and gives both device families equal weight. Reported seed intervals resample entire recordings 4,000 times.

| Model given the correct speech | Coffee / computer, three seeds (%) | Printer / camera, three seeds (%) |
| --- | ---: | ---: |
| SLWM | 16.6 / 14.7 / 11.5 | 11.5 / 11.7 / 12.7 |
| Dense GPT-2 | 12.5 / 12.5 / 12.2 | 9.9 / 12.4 / 11.2 |
| Gated recurrent | 12.6 / 14.0 / 11.1 | 9.5 / 12.5 / 14.1 |
| Most frequent training action | 23.4 | 18.1 |
| Repeat the previous completed action | 11.1 | 16.5 |
| Training transition after the previous completed action | 19.7 | 22.3 |

Replacing the spoken cue changed the seed-1001 hybrid's predicted verb on 51.2% and 60.5% of decisions in the two folds. The corresponding accuracy gains from the true cue were +0.8 and −0.9 percentage points. Sensitivity to language can therefore coexist with weak grounding. Every architecture also has independently fitted deranged-supervision and observation-only controls. Closed-vocabulary exact verb–noun scores count held operands that never appeared in optimization as misses. [All conditions, baselines and paired intervals](metrics/primary.json) · [Raw paired predictions](metrics/primary-predictions.npz)

The speech channel combines an automatic transcript with features of the actual instruction audio. Speech overlapping the matched historical microphone prefix is suppressed to prevent the original words leaking into a wrong-instruction trial. Held word error was 67.4% / 53.8% with Whisper tiny.en and 49.6% / 34.1% with small.en. Replacing the transcript or providing acoustics alone is scored separately. The human transcripts provided an additional clue: relative to the wrong directive, the chance of explicitly mentioning a verb among the next three annotations rose by 6.1 [0.7, 11.5] and 8.1 [3.3, 12.5] points. For the immediate verb alone, the changes were 3.5 [−0.5, 7.4] and 4.1 [about 0, 7.6] points. This is an exploratory lexical check, sensitive to the fixed word groups. [Recognition and timing measurements](metrics/measurement.json)

## Physical effect

The target is the fraction of stable depth cells that move more than 25 mm in a world-registered local patch around the manipulation. The separate magnitude target uses displacement in millimetres. Four distinct measured views bracket the action. We exclude tracked hand geometry and compare with neutral pre-action fluctuation, a wider surrounding patch, no-change persistence and training constants. The synthetic camera-motion fixture measured zero change on a static plane; it recovered exactly 50% changed cells when half of a plane moved 50 mm.

Of the sampled held actions, 106 changes in 39 coffee/computer recordings and 153 in 41 printer/camera recordings met the numerical stability criteria. Correct-minus-wrong improvements in hybrid surface-fraction MAE were +0.59 / +1.09 / −1.05 points in fold A and −0.30 / −0.22 / −0.31 points in fold B. No hybrid, dense or gated seed passed the two-instruction-contrast physical gate. No-change persistence had 15.8 and 9.0 points of MAE, while the training mean had 13.0 and 11.3. The complete results also include original-scale magnitude, nearer and farther surfaces, and isolated-action sensitivities.

Separate blinded prescoring found 12 clear device centers in the initial 24 cases. Reviewing every held pre-action center identified **156 of 259** as clearly on a device or component, including 7/22 coffee, 60/84 computer, 56/87 printer and 33/66 camera centers. The foreground-minus-surrounding fractions were −1.8 [−6.5, 3.5] and −0.2 [−1.9, 1.6] points. These measurements establish local surface change with limited attribution to the particular device. They do not capture internal power, contact force or the success of a hidden mechanism. Device-centered and isolated subsets are reported separately. [All blinded grades](metrics/device-audit.json) · [Geometry conventions](measurement.md)

The primary study completed 54 fits and verified 162 adapter/optimizer generations on two physical Fedora disks. Every output reads a 128-coordinate shared field. The matched transfer has 507,204 trainable parameters per model. The larger equal-exposure experiment is a separately registered comparison. [Checkpoint identities](metrics/primary-artifacts.json) · [Scale protocol](scaling.md)
