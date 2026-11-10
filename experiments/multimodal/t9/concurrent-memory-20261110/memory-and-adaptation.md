# Full history and decision adaptation

The two follow-ups use distinct new 128-mission cohorts. Each mission lasts 51.2 seconds and requires eight consecutive goals. The full-history study changes only inference. The decision-adaptation study holds the backbone and all sensor decoders fixed.

| Native model, 512 emitted coordinates | Native 64-step attention | Full 512-step attention | Reset every layer every 32 steps |
| --- | ---: | ---: | ---: |
| SLWM, seed 911 | 94.4% | 94.9% | 94.6% |
| SLWM, seed 912 | 90.2% | 90.6% | 94.4% |
| Dense GPT-2, seed 911 | 36.9% | 36.2% | 90.2% |
| Dense GPT-2, seed 912 | 33.8% | 32.6% | 83.3% |

The metric is time at the requested goal. Giving dense attention the entire causal prefix did little. Timing its resets to match the 32-step training sequences greatly improved control in both seeds. This supports a specific advantage for the hybrid during uninterrupted operation and identifies a strong, inexpensive dense inference control. Both architectures should next be trained on variable-length 32–512-step contexts before attributing the remaining difference to the temporal operator. [Per-mission scores and cache sizes](metrics/followups/horizon)

On the separate decision cohort, a small head fitted to 16,384 expert-labelled states visited by the parent reduced occupancy in all four large wide-field parents. For SLWM the two seeds went from 94.5% / 90.6% to 89.5% / 65.2%. Dense GPT-2 went from 39.6% / 33.6% to 30.7% / 23.1%. Fitting the same head on randomly collected trajectories gave 94.5% / 91.3% for SLWM and 37.9% / 28.5% for dense GPT-2. This one-round procedure produced no replicated gain over the parents. The archived heads and per-mission measurements make the result directly replayable. [Trained decision heads and controls](metrics/followups/heads)

A 51.2-second replay shows the first registered held mission for the large 128-coordinate seed-912 pair. The black marker follows each model's issued control under the simulator's held stall dynamics. The colored outline indicates the requested station. The physical controller sees the same sensor values and issued action history. [Watch the held mission](examples/held-live-control.mp4) · [Exact action stream](examples/live-trace.json)

The external contact bank clips force inputs at ±20 N. In held robot scenes, 18.4% of clock bins contain a clipped force component and 6.1% of all force components are clipped. The maximum observed magnitude is 49.21 N. The registered score uses the bank range. A second readout restores the original timestamped force targets for a sensitivity check. [Measured force range](metrics/force-range.json)
