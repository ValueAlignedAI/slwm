# G2 cache-precision erratum,2026-11-09

All12 G2 cores finished12000 updates. Before any final lock or final-observation
scoring, the frozen-readout fitter failed: MHealth physical-invariant FFT-power
features were finite in FP32 but overflowed the FP16 cache. The training cache
had5716 nonfinite values and selection1766. No learned representation cache
contained nonfinite values.126 readout heads had completed; no final metric had
been computed. The original failed files, sources, heads and logs are retained.

The explicit v2 evaluator changes storage precision only for the physical-invariant
control, to FP32 consistently for HAR and MHealth in all folds. Feature formulas,
training-only normalization/SVD, labels, examples, learned representation caches,
head capacity, seeds, optimization, selection and metrics remain identical. The
HAR invariant heads are refitted after recaching; all other completed heads are
retained. The original49 core/evaluation source files and trained checkpoints
remain immutable; separately named v2 modules, their source snapshot and hashes
record the correction. Final lock includes the erratum and v2-source hashes.

This is an engineering correction before final tests, not an outcome-based model
selection or a changed core-training recipe. A lower physical-control error may
make the registered transfer gate harder. No final observation may be opened
until all repaired readouts freeze. No inference weight updates are introduced.
