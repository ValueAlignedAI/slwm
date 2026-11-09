# What survived the shared bottleneck?

As of 9 November 2026, all ten 12-layer variants have completed their first 100 joint updates on the RTX 3090. Each seed used identical input tensors across all five variants, verified by rolling content hashes. Each checkpoint saw 6,400 examples during this screen. The longer 1,000- and 8,000-update comparisons were fixed in advance.

## Why G2 struggled

G2 trained its reconstruction heads to recover the entire observed window from its final 512-value recurrent state. On every robot batch the hidden numeric mask was empty. The 0.9-weighted masked reconstruction term therefore contributed zero. The 0.1-weighted visible reconstruction term still supervised past commands, averaged over channels and time. Future numeric supervision contained pose and gripper state. This gave the core little direct incentive to preserve the most recent command.

A linear probe trained on 29,332 G2 robot windows and scored on three separate selection groups supports that explanation. For the two large signal seeds, using all 512 dimensions gave mean group-and-coordinate R² of −0.060 and −0.068 for the previously issued command. Using 128 dimensions gave +0.009 and −0.031. The matched Transformers scored +0.014 and +0.039 with all dimensions. Untouched random cores scored +0.118 and +0.128. A training-mean predictor scores zero under this metric. These values describe linear accessibility. An ablation of the training objective would distinguish its contribution from the recurrent compression.

## What changed in G3

Each modality now passes through the same sequence of shared states. Direct current and next-position targets supervise each signal position. Both model families reuse the actual matched 500,006,912-token language parents. The frozen language backbone carries joint inputs, and learned residual adapters update inside all twelve blocks. General next-token prediction takes 20% of new updates. The full budget per run is 512,000 example presentations, including 9,830,400 language targets. The 118,272 eligible COCO training images and 50,000 new unlabelled CIFAR-100 training images will each be traversed completely. The speech pool contains 50,193 eligible utterances. The much smaller pools of 932 eligible robot episodes and six PAMAP2 recording rows limit claims about physical scale.

Dense GPT-2 stays dense at both adapter widths. The narrow dense models have 129.0M to 129.7M total weights, with 4.58M trainable. Sparse SLWM has 140.36M total and 15.23M trainable. Its nonexpert plus one-expert-per-layer count is 129.75M. The wide dense GPT-2 and SLWM controls have 139.64M and 140.33M total. This compares active adapter width and stored adapter capacity separately. A separately registered six-layer experiment tests deployed backbone depth and inherits the same language parents.

## Can G3 retain the command it actually received?

A subsequent diagnostic sampled up to four windows per old robot episode and kept G2/G3 case IDs and previous-command targets identical. Each ridge probe used 1,024 fit windows. Scores average across seven command coordinates and three held-out selection groups containing 300 windows. These are retrospective data. G2 had its original 32-tick multiview and goal inputs. G3 saw the last 24 ticks, one camera and numeric history.

| Frozen field at the final observed tick | Seed A R² | Seed B R² |
|---|---:|---:|
| G2 large signal model | −1.009 | −0.771 |
| Unadapted G3 signal backbone and new edges | 0.844 | 0.804 |
| Dense G3 signal, 100 joint updates | 0.928 | 0.936 |
| Same procedure with historical command channels erased during fit and scoring | 0.026 | −0.276 |
| Sparse-adapter G3 signal, 100 joint updates | 0.903 | 0.932 |
| Matched dense G3 GPT-2, 100 joint updates | 0.771 | 0.724 |

The new interface already made the observed command accessible at initialization. Joint updates added roughly 0.08 and 0.13 R² for dense SLWM. Erasing the actual command history removed most of that recoverability. The fresh Push-T endpoint test below asks the harder question of whether these features help predict an external consequence.

## Fresh-task transfer

These development accuracies use 32 labels per class. The digit benchmark has a single development speaker. Its group bootstrap has only one possible resample, so it gives no speaker-population precision estimate.

| Fresh task | G2 dense SLWM, two seeds | G3 dense SLWM, two seeds | G3 dense GPT-2, two seeds | Raw input |
|---|---:|---:|---:|---:|
| Spoken digits | 19.2%, 15.2% | 22.8%, 30.6% | 25.2%, 36.8% | 28.8% |
| Fashion images | 58.7%, 63.6% | 70.2%, 70.4% | 68.8%, 69.9% | 75.3% |
| Banking intents | 5.7%, 6.6% | 70.1%, 70.3% | 67.1%, 68.2% | 67.7% |

The unadapted signal parents already scored 69.8% and 70.4% on banking intents. This makes language transfer a strong explanation for the improvement over G2. The two trained sparse models scored 29.4% and 29.6% on spoken digits. They traded places with dense GPT-2 across seeds. The original three-task gate passed for none of the signal variants at this budget.

Routing differed by source. Averaged across layers, the exponential entropy of the four expert loads was 3.88–3.89 for text and 2.26–2.27 for robot observations. This measures specialization of dispatch. The separate held-out scores measure transfer quality.

## Fresh physical prediction

Using only 32 demonstration episodes to fit each readout, we predicted object displacement eight steps ahead on 64 unseen Push-T episodes, totalling 512 held-out windows. The readout received the recorded plan alongside frozen features. Lower error is better in the native 512-unit arena.

| History and plan | Mean endpoint error |
|---|---:|
| Zero displacement | 13.90 |
| Raw image and pusher history, recorded plan | 13.38 |
| Dense GPT-2, two seeds | 13.85, 14.10 |
| Dense SLWM, two seeds | 13.91, 14.03 |
| Sparse SLWM, two seeds | 13.94, 13.89 |

The raw readout scored 14.09 when the plan was withheld and 18.27 when plans were permuted across episodes. The recorded plan contains predictive information. The early learned representations have yet to extract a physical prediction advantage over the raw control. For dense signal seed 81 the paired episode error reduction against raw was −0.53 with a 95% interval from −0.95 to −0.14. All ten configurations missed the physical-transfer gate.

The 8,000-update physical confirmation uses a separate 94 episodes. The three-task final evaluation uses official visual and text test sets and two different spoken-digit speakers. Model selection and threshold changes on these outcomes would require a new study.


## Sparse backbone and deployment checks

The additional 2-of-4 experiment removes 28,311,552 weight positions across both feed-forward matrices in all twelve shared SLWM blocks. Dense GPT-2 references retain all their weights. On the 3090, the first packed FP16 matrix at 1,536 input positions agreed with the identical zero-masked dense matrix to relative L2 error 0.000021. Its median time was 0.458 ms, compared with 0.279 ms for dense execution. At this shape the hardware kernel made inference slower. Training uses dense kernels on the fixed masked weights.

All four original language parents, four six-layer parents and two structured parents were SHA-verified on the Mac. The ten initial 12-layer joint deltas were verified separately. A sparse-adapter checkpoint loaded independently for both text continuation and next-audio prediction from a real held-speaker observation. These checks establish artifact usability.
