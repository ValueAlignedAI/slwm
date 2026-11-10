# A1 implementation revision2

The first attempt stopped in its initial selection diagnostic before any optimizer
update: autocast converted the covariance product to BF16, unsupported by the
CUDA eigensolver. Preserve its source snapshot and failure log in `angular_pilot`.

Revision2 uses the identical512-update parents, data, objectives, coefficient,
optimizer reset and success criteria from `angular_alignment_pilot_20261106.md`.
It disables autocast only for covariance/geometry/regularizer calculations;
model forwards remain BF16. The revision is registered before any optimizer
update and saves a separate `angular_pilot_v2` artifact root. No calibration or
final prediction informed the fix. G1 training sources/weights remain unchanged.
