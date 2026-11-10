# G3 retained-depth comparison

The 12-layer study varies adapter parameters while holding backbone depth fixed. This secondary study compares 6 against 12 actual executed layers, within dense GPT-2, dense SLWM and sparse-adapter SLWM at two seeds. It was registered after the 100-update primary development scores and before the depth study or the primary confirmation set was opened.

Retain the second block of every successive pair from each verified 500M-token parent, then renumber blocks. Retain original word/position embeddings and final normalization. Every deployed backbone weight belongs to the smaller model. The four reduced parents inherit language pretraining from the full-depth teacher; the experiment measures the cost and transfer quality of depth reduction during adaptation. It cannot estimate independently pretrained small-model scaling.

Use the original exact source schedule, 100, 1,000 and 8,000 updates per profile, training objectives and 1/8/32-shot test recipe. Compare the paired content hashes at every budget. Score the smaller zero-update parents too. Defer the longer reduced-depth runs until the primary matrix completes. The secondary confirmation lock binds the six reduced endpoints, full-depth endpoint cards, transfer evaluation sources and depth protocol. Publish both depth outcomes even when some controls outperform the joint model.
