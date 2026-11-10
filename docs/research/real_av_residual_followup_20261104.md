# T2 residual forecast follow-up — frozen before new test evaluation

## Discovery result and one targeted change

The completed absolute-feature cycle (`EXP-T2-201`–`209`) passed the audio
prediction and hybrid-versus-vanilla gates in both seeds (7–8% audio MSE gains),
but failed reliable cross-modal contribution. Video error was worse than the
simple last-observed feature. That cycle is a discovery set for this decision;
its test people are not reused as the follow-up confirmation set.

Test one change: each modality head predicts a residual added to its last
observed feature (zero/train mean if that modality is absent). Initialize
residual heads to zero, so the initial forecaster exactly equals persistence.
There are no extra trainable parameters. All input masks and blind retrieval
rules remain. This tests whether spending small-model capacity copying static
appearance was obscuring learnable dynamics; it does not prove general physics.

## Fresh test and unchanged preprocessing

Train/validation clips, feature files, PCA and normalization are byte-identical
to the discovery corpus. Download 30 balanced clips each from seven completely
unused people: `1027, 1018, 1073, 1061, 1003, 1063, 1077`. They were selected
with seed 20261105 from people outside all three discovery splits, without
examining their clips or scores. Hash and verify both original training files
and the frozen normalizer. Future audio windows remain midpoint-safe.

## Frozen budget, matrix and gates

Use `configs/t2/cremad_residual_matrix.json`, 600 × 32 clips per arm. Run full,
vanilla, no spectral, no long convolution, shuffled-pair training, audio-only,
video-only, and second-seed full/vanilla, IDs `EXP-T2-211`–`219`.
The three original gates are unchanged: ≥5% improvement over the stronger null
in at least one modality; ≥2% paired-other contribution over a within-person
wrong other stream with positive person-cluster 95% interval; ≥5% error gain or
≥5 percentage point independent retrieval gain over vanilla in both seeds.
All arms now consume an identical sample/dropout RNG schedule. The first cycle's
single-modality arms consumed different sample orders (same sample budget), an
execution limitation to retain in the discovery report.

An additional prespecified diagnostic is whether residual prediction beats
video persistence by ≥5% on fresh test people in both full-model seeds. Failure
rejects this residual recipe as a useful video dynamics model at this budget.
Do not select a new architecture, threshold, or training budget from fresh test
results. If the multimodal or dynamics gates fail, stop this recipe and write
the result; do not start joint text/code training or a larger model.

The original cycle sources are archived under its `source_snapshot/`; the
follow-up records its own source/config hashes and snapshots. Original
checkpoints retain the default absolute-head behavior when reloaded.
