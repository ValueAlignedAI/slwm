# T2-V1 output wrapper correction, evaluation revision2

The eleven registered training runs, validation-only choices and final feature
evaluations completed without changing their frozen sources. The first audio
gallery attempt then failed on its first case:207 fine audio frames cannot
reshape into52 video packets of four frames. CREMA preparation already permits
up to three terminal missing subframes and repeats the last analyzed frame.

Apply that existing edge rule to ordinary-file inference and the output wrapper.
The repeated frame is calculated from the globally erased recording; it cannot
restore hidden evidence. No targets, weights, hyperparameters, test scores or
gates change. Add a real short-tail waveform test. Preserve the failed log,
partial output folder and original evaluation source snapshot. Revision2 has its
own source snapshot/hash manifest and renders the same32 previously fixed IDs.
Resume only audio rendering, external-recognizer evaluation, audits and reports;
do not rerun or select the already completed feature comparisons.
