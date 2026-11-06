# A1: bounded angular-collapse repair

Exploratory selection-driven repair, registered before its optimizer updates.
At G1 update512, several image/sound/text alignment projections have nearly
parallel unit vectors and covariance participation near one. Raw feature
variance can be satisfied through amplitude changes without useful angular
information. This is early evidence of representation collapse, not proof of
task memorization or a final result for G1. The registered G1 matrix stays
unchanged and its final roles remain closed.

Snapshot matched512-update G1-101/102 inference parents, with exact hashes.
For each processor compare the original objective against that objective plus
an angular variance/covariance/mean penalty, coefficient.2. Both reset AdamW,
start from the same within-processor parent, use exactly512 alternating COCO/
Clotho training calls and the same examples/order, LR1e-4, weight decay.1,
batch sizes256/64, BF16 and norm clipping1. Neither arm sees final/calibration
data. This focused diagnostic does not replace the heterogeneous world study.

Regularize independently encoded signal and caption unit vectors. Multiply by
sqrt(128), penalize per-feature standard deviations below.5, mean squared
off-diagonal covariance, and the norm of the batch mean unit vector. The
original raw-variance term remains in both arms. Both compute the same extra
forward/regularizer graph, with coefficient zero in the control. This follows
the variance/decorrelation motivation of [VICReg](https://arxiv.org/abs/2105.04906),
but is a normalized-vector adaptation rather than a replication.

Every128 updates report fixed selection record retrieval, angular geometry and
masked reconstruction. Keep endpoint weights, parents, curves, exact source/
data hashes and actual compute. A useful pilot needs better paired retrieval
on both source families and both processors, participation ratio>=8, and at
most10% worse fixed-target masked error than its reset-optimizer control.
Do not select a winner from rank alone or call this new-source generalization.
If geometry improves without retrieval, distinguish collapse prevention from
learning a shared semantic model. A changed main objective needs a new study
and fresh confirmation, rather than silently modifying G1's in-progress run.
