# Is the real-contact decline consistent with overfitting?

The first terminal models lost held contact AP even as their action errors fell. This post hoc diagnostic is registered before checking any training-set contact AP. Compare the 2,048- and 8,192-update checkpoints for both sizes, both cores and both seeds. Select up to 30 training windows per original recording date with a fixed label-independent RNG, using the same rows for every checkpoint. Evaluate the original 668 held development windows and identical 8–63 tick restriction. No weights or calibration parameters change.

Report contact prevalence, model AP, raw motion AP, gain over motion, per-date gains, Brier score and the class-weighted binary loss under the original training odds. Distinguish a growing train-versus-held ranking gap from a decline on both sets or a mismatch between ranking and the trained objective. Training windows are in-sample. The held real dataset has only three recording dates.
