# Speech grounding and real actuator calibration

A frozen audio encoder changed what the small model could recover from an unfamiliar voice. Both systems used the same 182,535-parameter speech head, the same 12,704 training utterances from 16 speakers, and the same development rule. The second system added 88,154,112 externally pretrained Whisper encoder parameters. It recovered the spoken color, letter and digit on 96.3% of complete utterances from two newly acquired people, compared with 54.6% for the trained acoustic features. The eight older held speakers scored 98.1% and 64.5%, respectively. These numbers weight each voice equally. Training, calibration and evaluation speakers are disjoint for the locally trained speech head. The upstream Whisper pretraining membership of these public recordings is unknown. The two newly acquired speakers scored 92.6% and 100% with the pretrained encoder. The comparison supports reusing pretrained acoustic knowledge. The architecture comparison gives both candidate physical processors exactly the same encoder. [Speech metrics and data partitions](metrics/speech.json) · [Frozen choice](speech-lock.json)

A public station map disambiguated every one of 256 scheduled commands from the two new speakers. The threshold estimated on 64 calibration-quartet decodings admitted 215 and rejected 41 correctly understood commands. Its threshold was 0.9934. The deployment comparison preserves this rule and includes a geometric-control diagnostic with the confidence restriction opened. Scheduled commands include repeated recordings across scenes. Acoustic accuracy alone cannot tell us how many selective placements will succeed. The first recognition study had exposed the original encoder's held scores before we introduced the frozen encoder. Its weights, its selection rule and all competing physical policies were developed independently of the held physical trajectories. [End-to-end design](design.md) · [Development record](development-decisions.md)

## A small amount of real telemetry changes the answer

At 5 Hz we fitted a 0.8-second effector-response prediction from 3,264 simulated windows on 192 starts. All simulator targets came from executed dynamics. Its command and pose distribution used the real training split. We evaluated two previously released groups of genuine recordings, spanning 196 and 210 independently grouped parent recordings, respectively. All real calibration examples came from a separate training population.

| Training information | Four-step error on 196 parents | Four-step error on 210 parents |
|---|---:|---:|
| Assume no movement | 93.81 mm | 83.81 mm |
| Integrate recorded offsets | 37.55 mm | 33.40 mm |
| Fit simulator response | 34.09 mm | 30.18 mm |
| Calibrate with 1 real training parent | 12.07 mm | 11.03 mm |
| Calibrate with 4 real training parents | 5.55 mm | 5.14 mm |
| Calibrate with 16 real training parents | 4.94 mm | 4.58 mm |
| Fit all 583 real training parents | 4.64 mm | 4.32 mm |

At one real training parent, simulation initialization saved about 0.15–0.16 mm over fitting directly on that same parent. At four parents, real-only training reached 5.55 and 5.14 mm. Most of the improvement came from observing the physical actuator. Each few-shot result averages five calibration orders, with uncertainty clustered by real test parent. The experiment measures actuator transfer on previously released real recordings. Independent alternative tabletop placements are executed in the simulator. [Full paired uncertainty](metrics/telemetry.json) · [Transfer protocol](telemetry-transfer.md)
