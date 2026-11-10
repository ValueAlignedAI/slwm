# T1-S5: matched500M-token continuation, registered2026-11-06

Hypothesis: the early equal-token signal-path advantage survives greater token
exposure and improves raw continuation stability. Compare both architectures and
seeds81/82, preserving their own full S4 selected optimizer/RNG states. Parents
221–224 each consumed150,994,944 tokens. New IDs231–234 use500,006,912 total
tokens: the first full32768-token optimizer batch at or above500M. This is10,651
additional updates/model; there are no repeated logical training windows.

Keep original model/512 context, immutable2.6B-token FineWeb-Edu corpus, batch64,
BF16, tied Liger0.8.5 CE, AdamW/betas/weight decay and clipping. All four arms use
the same token-count LR schedule: smoothly restart2e-5→1e-4 across256 updates,
then cosine to1e-5. This is a registered continuation recipe, not uninterrupted
original pretraining. Preserve parent files/hashes; save atomic best/resume/final
states and inference milestones near250M/375M/500M, with actual counters.

Interleave bounded256-update slices through the existing GPU lock, allowing
independent multimodal experiments. Log training/validation time separately from
checkpoint loading/saving and queue waits. Maintain equal maximum token budgets;
selected weights may differ in exposure and must be labeled. Stop an arm if two
successive validation checks exceed its best NLL by5%; do not conceal a target
shortfall or call unequal endpoints token-matched. Do not change sources/config
after source_guard creates the snapshot; new runtime revisions need new IDs.

Use the existing domain-disjoint validation stream for selection only. Wait
until all four complete/freeze before any new final test. Evaluate both fixed
endpoints and validation-selected states on1,024 previously uninspected test
windows starting at logical4096 (prior S2/S4 accesses0–255 and512–1535). Score
their own151M parents on this same new slice after fitting. Primary architecture
comparison is the500M fixed endpoint, conditional on both reaching the target.
Also report selected NLL/PPL, actual exposure, time/quality curves and repeated
raw greedy/sampled continuations. Parent and selected scores are diagnostics,
never substitutes for a failed fixed-endpoint comparison. Test results never
select weights, schedules or decoding temperatures.

Strict neural accounting:124,439,808 GPT-2-style and125,131,008 SLWM weights,
including tied embeddings. Both are random-init lineages, separate from any
multimodal world model. Report both seeds, all failures and repetitive samples;
neither lower text NLL nor longer generation establishes world knowledge or
physical-action competence. The estimated additional active GPU budget is11h,
with multimodal GPU work, I/O and final evaluation additional.
