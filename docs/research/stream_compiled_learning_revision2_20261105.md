# Compiled learning pilot, runtime revision2

This is an explicit continuation of the unchanged
[original pilot protocol](stream_compiled_learning_protocol_20261105.md).
The original source/config/logs and200-update SLWM optimizer state remain in
`compiled-learning-20261105`. It stopped before the first Transformer update
when the shared `process` code object exceeded Dynamo's default eight cached
variants across models/shapes/gradient modes. No fresh test data was read.

Use new IDs841–844 and directory `compiled-learning-v2-20261105`. Reset Dynamo's
in-memory function cache before each model construction, preserve its on-disk
Inductor/Triton kernel cache, and explicitly allow at most32 compiled variants.
This runtime-cache revision does not alter tensors, objective, data order,
optimizer, LR horizon, precision or400-update endpoint. Fullgraph compilation
continues to fail loudly rather than silently falling back to eager execution.

Resume saved831 step200 model/optimizer/RNG/curves into841, recording original
file hashes and lineage. Other arms start from their paired initializations.
Keep all original failed-run files. Eager reference843/844 and compiled841/842
share the exact400-update schedule. Retain the original1% fixed-endpoint
validation check and all component losses. No extra primary B1 or policy test
candidate is introduced. No test speaker is used for the pilot.

Pilot timers include first compilation/cache loads occurring inside its slices,
but reuse a persistent disk cache warmed by the earlier probe. Report the
additional original79.29s/37.63s SLWM/Transformer probe startup separately; the
inherited `cold_inclusive_speedup` field must not imply a brand-new empty cache.
The probe's model-step timings exclude common CPU batch construction and are
not end-to-end speedups. At its measured step savings, original cold startup
alone amortizes after roughly507/337 updates, with other preparation/validation
costs additional. No claim of preserved long-run or fresh-person quality follows
from this single-seed400-update diagnostic.
