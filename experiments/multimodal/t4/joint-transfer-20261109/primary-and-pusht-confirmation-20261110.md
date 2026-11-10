# G3 primary transfer and Push-T confirmation

**10 November 2026.** All ten full-depth models reached 8,000 updates with matching input hashes within seed. Each saw 512,000 multimodal example presentations after its 500,006,912-token language pretraining. None of the three signal variants passed the registered requirement for superiority on two tasks in both seeds against the matched dense Transformer, raw features and unadapted parent controls. Only the wide signal variant beat all its controls on Fashion in both seeds. [Registration](../../../../docs/research/joint_transfer_20261109.md) · [Full primary results and paired intervals](summary-8000-test.json)

## What adaptation transferred

The table gives balanced accuracy with 32 labelled support examples per class. Paired entries are seeds 81 and 82. Confirmation used 1,000 spoken-digit recordings from two held-out speakers, 10,000 Fashion images and 3,072 Banking77 queries. [All ten models, full control table and training times](summary-8000-test.md)

| Frozen readout input | Spoken digits | Fashion images | Banking intents |
| --- | ---: | ---: | ---: |
| Raw features | 66.4% | 74.5% | 69.3% |
| Unadapted dense signal parent with initialized edges | 44.1% / 49.5% | 69.6% / 69.5% | 70.0% / 70.5% |
| Adapted dense GPT-2-style Transformer | 54.0% / 47.3% | 75.7% / 75.8% | 67.3% / 68.9% |
| Adapted dense signal model | 61.2% / 55.4% | 76.2% / 75.5% | 69.4% / 69.8% |

Dense signal adaptation improved spoken-digit accuracy over its parent by 17.1 and 5.9 percentage points and over dense GPT-2 by 7.2 and 8.1 points. Raw audio features still scored 5.2 and 11.0 points higher. For Banking77, dense signal adaptation lost about 0.65 points relative to its own parent at each seed. The dense signal models retained lower text NLL than dense GPT-2, although NLL rose by 0.019 and 0.024 relative to their own parents. On 256-way paired retrieval their COCO top-one scores were 7.03% and 6.64%, and Clotho scored 1.56% and 1.17%, against 0.39% chance. These results support limited alignment and task-specific transfer.

## Recorded action-conditioned forecasting

Push-T used a separate fixed ridge readout with 32 fit episodes. The two held sets were mutually disjoint. Error is mean within-episode Euclidean displacement error over an eight-step forecast in the 512-unit arena. Lower is better.

| Held set | Episodes | Zero displacement | Raw joint input and plan | Dense GPT-2 and plan, seeds 81 / 82 | Dense signal and plan, seeds 81 / 82 | All learned joint models |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 100-update screen | 64 | 13.90 | 13.38 | 13.85 / 14.10 | 13.91 / 14.03 | 13.75–14.15 |
| 8,000-update confirmation | 94 | 14.58 | 14.24 | 14.45 / 14.51 | 14.34 / 14.45 | 14.29–14.62 |

Supplying the recorded plan improved dense signal forecasts relative to withholding it at both seeds. Raw history with the same plan had lower mean error than every learned joint model at both budgets. None passed the registered 5% improvement and paired-interval gate. This experiment supports sensitivity to plan input. Closed-loop decision quality requires an intervention test. [Screen](pusht/screen/results.json) · [Confirmation](pusht/scale/results.json)

## Companion comparisons

All results below use the repaired, hash-matched data schedule. The six-layer dense signal model scored 47.35% on development Banking77 in both seeds at 100 updates. Its six-layer parent scored 43.31% / 43.04%, full-depth dense signal scored 70.06% / 70.27%, full-depth dense GPT-2 scored 67.08% / 68.16%, and raw features scored 67.71%. [Six-layer screen](depth6/depth-100-dev.json)

At 1,000 updates, the 2:4 backbone scored 25.0% / 44.2% on development digits, 74.6% / 74.7% on Fashion and 70.0% / 69.8% on Banking77. Its scores were within 1.2 percentage points of full-depth dense signal on all three tasks at both seeds. Both companions still require their registered 8,000-update confirmations. [2:4 screen](structured/structured-100-dev.json) · [2:4 development comparison](structured/structured-1000-dev.json)

The independent audit verified all ten final primary inference deltas and thirty fitted readout heads against their registered files, the second HDD and Google Drive. It also verified every published companion milestone, the four reduced language parents, the two structured language parents, and both Push-T episode and endpoint locks. [Artifact and episode audit](primary-and-pusht-audit-20261110.json)
