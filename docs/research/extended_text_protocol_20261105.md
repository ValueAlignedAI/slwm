# T1-S3: longer matched text learning

Registered before additional updates. Continue all four T1-S2 selected step-1024
weights AND AdamW moments, with the same paired unique-window schedules starting
after the 16,777,216 already consumed tokens. Never overwrite the parent runs.
Add 4,096 updates of 32×512 tokens: 83,886,080 total tokens per completed run.
This is still far from training on the prepared 2.6B-token corpus.

Use a declared warm restart, shared by both architectures: linearly move from
6e-5 to 2e-4 over 128 updates, then cosine to 2e-5. It is not an uninterrupted
original cosine run. Restore optimizer/RNG states; retain all counts/hashes.
The corpus, tokenization, 512 context, architecture and paired data stay fixed.

Validate on the same fixed 131,072 reserved validation tokens every 512 updates.
Stop after two checks more than 5% above the best previous validation NLL; report
any unequal exposure. Select with validation only. After all four runs freeze,
evaluate 524,288 tokens from test windows 512 onward, disjoint from the 256 test
windows inspected in T1-S2. This is a fresh test sample, not new source domains.
Report NLL/PPL, train/validation gap, actual exposure, both seeds, memory and wall
time. A lower loss at equal tokens is not equal-FLOP efficiency. Save all fixed
prompt continuations without selection. No code/reasoning/world-model claim.

GPU slices release the shared research lock every 512 updates, so speech and
operational experiments can share the single 3090 without concurrent training.
Each slice saves a full optimizer/RNG resume state. Preserve the original and
extended best checkpoints. All training source/configs are frozen before launch.

Primary analysis: whether the paired SLWM/GPT-2 NLL ratio persists with five times
the original token exposure, with validation improvement in both seeds and no
overfit stop. Report absolute changes even if that question has a negative answer.
The corpus scale is adequate for this trial; consumed data still severely limits
any extrapolation to convergence or a unified multimodal representation.
