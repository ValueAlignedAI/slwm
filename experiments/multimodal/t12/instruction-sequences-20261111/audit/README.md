# Observable device-state evidence

A single Codex visual reviewer classified 745 blinded pairs, including 57 hidden repeats of 688 distinct cases. The panel mixes four views per case. Two frames precede the instruction cutoff by 0.56 and 0.12 seconds. The other two fall 0.23 and 0.92 seconds after the latest ending action in the triplet. The two sides were randomly exchanged. The reviewer saw no instructions, verbs, prediction scores, task labels, recording names or split assignments. The numbered review sheets, individual codes and frame times allow another reviewer to repeat the exercise. [Blind frame sheets with data license](frames.zip) · [Original ratings](blind-ratings.json) · [Source clocks, components, repeats and final labels](../metrics/visual-labels.json)

A change required a visible device configuration, such as an open toner bay, a memory module outside its socket, a mounted camera lens or a changed display. An identifiable comparable configuration in both views could be rated stable. Incomparable viewpoints, occlusion, ambiguous attachment and loose accessories received an indeterminate rating. Every changed rating names the observed component. Sampling balanced annotated manipulation likelihood and device family and capped each recording at four cases. This concentrates the state study on readable manipulations and affects estimated prevalence.

The reviewer agreed on 56 of 57 repeats across the three outcomes. Cohen’s κ was 0.966. All 20 duplicate pairs judged readable in both presentations agreed on changed versus stable. This measures consistency of one reviewer. Independent human agreement and continuous mechanical or electrical telemetry remain open.

| Fresh family | Decisions with readable change | Readable stable | Readable total |
| --- | ---: | ---: | ---: |
| Coffee | 14 | 6 | 20 |
| Computer | 7 | 9 | 16 |
| Printer | 20 | 5 | 25 |
| Camera | 8 | 8 | 16 |

The fresh panel selected 144 cases and yielded 77 readable endpoints. Its two folds include 20 and 22 recordings with usable outcomes. The earlier replication panel yielded 71 out of 144. Unknown and conflicting cases never became no-change labels. Video and source data are from Wang and colleagues’ public HoloAssist release. [HoloAssist](https://holoassist.github.io/) · [CDLA-Permissive-2.0](../examples/HoloAssist-LICENSE.txt)
