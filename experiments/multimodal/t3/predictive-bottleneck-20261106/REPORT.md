# G2 predictive bottleneck and action-history findings

All fixed endpoints and controls are retained. No final outcome selected a source role, checkpoint or readout.

| Registered check | Outcome |
|---|---|
| all_fixed_budgets_completed | PASS |
| har_superiority_both_slwm_seeds | FAIL / insufficient evidence |
| actor_superiority_both_slwm_seeds | FAIL / insufficient evidence |
| har_raw_retention_both | FAIL / insufficient evidence |
| actor_raw_retention_both | FAIL / insufficient evidence |
| history_dynamics_signal_seed201 | FAIL / insufficient evidence |
| history_dynamics_transformer_seed201 | FAIL / insufficient evidence |
| live_replay_completed | PASS |
| registered_strong_transfer | FAIL / insufficient evidence |

## Frozen unseen-task transfer

| Task / run | Error | Strongest selection baseline | Paired gain95% interval |
|---|---:|---|---|
| har/activity / G2-201 | 0.1859 | raw-history | [-0.12710209935193012, -0.004445119216502327] |
| har/activity / G2-202 | 0.1603 | raw-history | [-0.08348615655018123, 0.011241527189547037] |
| har/activity / G2-203 | 0.1620 | raw-history | [-0.060936112574906685, -0.010087072672181287] |
| har/activity / G2-204 | 0.2715 | raw-history | [-0.20473953194075534, -0.08857462070146671] |
| har/activity / G2-205 | 0.1830 | raw-history | [-0.11303158342817367, -0.01091645779786718] |
| har/activity / G2-206 | 0.1611 | raw-history | [-0.07212683016108787, 0.0006544949318757178] |
| har/activity / G2-207 | 0.1195 | raw-history | [-0.02990142863145595, 0.043748607067463634] |
| har/activity / G2-208 | 0.1241 | raw-history | [-0.028507805456363666, 0.033655016314134846] |
| har/activity / G2-209 | 0.1759 | raw-history | [-0.11984632421054321, 0.01486937238727807] |
| har/activity / G2-210 | 0.1732 | raw-history | [-0.10037352412462099, 0.0006534286745140341] |
| har/activity / G2-211 | 0.1542 | raw-history | [-0.07061627639061967, 0.01155939520813661] |
| har/activity / G2-212 | 0.1992 | raw-history | [-0.1341104824751891, -0.023821276590185514] |
| droid/actor / G2-201 | 0.7390 | previous-command | [-0.6264061520497977, -0.5689688951843687] |
| droid/actor / G2-202 | 0.7520 | previous-command | [-0.6554214690678118, -0.5713407790914969] |
| droid/actor / G2-203 | 0.7661 | previous-command | [-0.6878622466644527, -0.5693622751131666] |
| droid/actor / G2-204 | 0.7368 | previous-command | [-0.6301043817087896, -0.5523956231308155] |
| droid/actor / G2-205 | 0.7522 | previous-command | [-0.6466425392444112, -0.5731177083579153] |
| droid/actor / G2-206 | 0.7389 | previous-command | [-0.6514904572073371, -0.550085675077396] |
| droid/actor / G2-207 | 0.7277 | previous-command | [-0.6361622110539267, -0.5456889939040025] |
| droid/actor / G2-208 | 0.7284 | previous-command | [-0.6195078763878362, -0.5523742732437321] |
| droid/actor / G2-209 | 0.7523 | previous-command | [-0.6652273065847184, -0.5686840371973497] |
| droid/actor / G2-210 | 0.7657 | previous-command | [-0.6810763430997123, -0.5693902973035201] |
| droid/actor / G2-211 | 0.7368 | previous-command | [-0.6291218002659402, -0.5578319215688908] |
| droid/actor / G2-212 | 0.7538 | previous-command | [-0.656612537123681, -0.5650919031769406] |

HAR intervals resample its nine official final people; actor intervals resample connected buildings/collectors. Raw/edge actor controls retain17 explicit physical values within the same128-input head budget. The primary actor receives only compact state. Random cores, all input interventions, secondary aligned-state probes and MHealth remain in `readouts/evaluation_summary.json`. Activity error is one minus macro recall; actor error is training-scale-normalized absolute error, with physical metrics also retained.

## Scale and cost

| Run | Parameters | Updates | Training+validation h |
|---|---:|---:|---:|
| G2-201 | 124,521,195 | 12,000 | 3.23 |
| G2-202 | 124,117,995 | 12,000 | 2.53 |
| G2-203 | 124,521,195 | 12,000 | 3.23 |
| G2-204 | 124,117,995 | 12,000 | 2.53 |
| G2-205 | 55,810,187 | 12,000 | 1.88 |
| G2-206 | 55,595,147 | 12,000 | 1.52 |
| G2-207 | 55,810,187 | 12,000 | 1.87 |
| G2-208 | 55,595,147 | 12,000 | 1.52 |
| G2-209 | 124,521,195 | 12,000 | 3.26 |
| G2-210 | 124,117,995 | 12,000 | 2.57 |
| G2-211 | 124,521,195 | 12,000 | 3.23 |
| G2-212 | 124,117,995 | 12,000 | 2.53 |

Both sizes use the same512-scalar state, source plans and per-seed effective example/packet budgets. Transformer references share the GRU posterior/dynamics so the primary pair isolates the field processor. Extra time is reported; no equal-compute claim or scaling law follows from two sizes. Bypass/point mechanism controls have one core seed each and require replication. The500M-token language models are independent weights; G2 text is supplied captions/goals.

## Dynamics, content and ongoing inference

The dynamics report retains1/4/8-step physical translation/SO(3)/gripper errors, normalized state MSE, image/background/motion errors, removed/shuffled plans and prefix shuffles with the final command fixed. Physical-second and ordinal-step velocities are separate. The registered mechanism gate is eighth-step moving physical-clock normalized pose MSE versus the matched point model, with5% relative improvement and a positive independent-group lower bound. Wrong plans have no observed counterfactual outcomes.

`alignment/summary.json` measures independent caption retrieval and the fixed unpaired sound→image categories against random/edge controls. It is not matched-event recognition. `speech_retention/summary.json` measures nonsilent hidden-phone error on core-unexposed voices, without an external vocoder; phone error is not waveform intelligibility. The older G1 study independently evaluates speech intelligibility and long raw media.

`operational/` logs persistent state, same-head reset controls, deadlines, stale rejection and exact weight hashes over more than40 seconds. Recorded observations do not respond to proposals. No physical device was actuated, and no reward/value function was learned. Data coverage failure remains explicit if no eligible native-time final episode exists.

## Usable artifacts

Each run retains `endpoint.pt`, validation-selected `best.pt`, a resumable optimizer/EMA checkpoint on Fedora, config/source/data hashes and full exposure logs. Inference snapshots sharing exactly the same bytes are hard-linked; no model state is approximated or compressed. `exports/<run>/actor.pt` contains the two frozen state-only heads and their training projections, tied to the endpoint SHA. Fixed HAR/robot observation/target examples are separate; no example is chosen by output quality.

```bash
python -m exploration.predictive_infer --checkpoint experiments/multimodal/t3/predictive-bottleneck-20261106/G2-201/endpoint.pt --observations experiments/multimodal/t3/predictive-bottleneck-20261106/exports/examples/droid_observations.npz --actor experiments/multimodal/t3/predictive-bottleneck-20261106/exports/G2-201/actor.pt --output /tmp/world_proposal.npz
```

Use the recorded code version/source snapshot. Observation examples are typed, normalized NPZ packets; raw physical inputs need the supplied normalizers and channel mapping. The `LiveWorld` API carries working memory and consumes only the latest encoded packet. Normal inference never changes weights.

## Next decision

The shared state does not meet the registered reuse/retention requirements. Retain task-specific gains and negative controls; inspect capacity, information loss, source imbalance and motion coverage before increasing parameters. A changed architecture needs fresh confirmation after these outcomes.

All data/source preparation failures, including the original protobuf decoder crash and the independent recovery audit, remain in the provenance record. Data counts and independent hours must remain distinct from repeated windows and camera tracks.
