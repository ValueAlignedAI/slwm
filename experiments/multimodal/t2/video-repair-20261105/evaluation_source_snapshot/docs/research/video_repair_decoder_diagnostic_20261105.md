# T2-V1 decoder diagnostic addendum

Fixed before final test scoring, while the registered training matrix runs.
The original protocol, model choices, training budgets and pass/fail gates stay
fixed. Add one explicitly labeled **oracle-spectrum** reconstruction to the
32-case listening/recognizer diagnostic. It supplies the true clean mel spectrum
to the same constrained phase decoder, whose phase is still initialized only
from observed waveform samples. This measures a decoder bottleneck separately
from missing-content prediction. It is an evaluation ceiling, not a trained
model, inference mode, or evidence of useful video. No oracle value enters any
model or its checkpoint selection. Keep this condition out of every primary
video/audio gate; retain source hashes with the render manifest.
