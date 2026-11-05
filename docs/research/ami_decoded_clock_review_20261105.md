# AMI decoded-clock review before fitting

The first training video pass stopped because its frame count was shorter than
the nominal AVI duration. Inspection of training ES2002a shows uniform decoded
timestamps beginning at0 for all checked cameras. Camera4 has31,194 decoded
frames ending at1247.72s, while its header declares31,212 frames/1248.48s; camera1
differs by one frame. Decoder warnings are retained. This is an endpoint-support
issue; no audio/video shift is inferred or fitted from test outcomes.

Measure each original video's complete decoded PTS sequence with one pinned
FFprobe runtime. Require zero origin and uniform cadence; a different clock
pattern stops preparation for review. Set the shared signal end to the earlier
of actual decoded support and audio support. Apply exactly the same rule to
every role. Clip only unavailable endpoints and their labels, keeping every
participant and the original clock. No unknown visual prefix is fabricated.

Cache source/timestamp hashes, decoded first/last times, nominal header duration,
decoder warnings and FFprobe version. Preserve original failed preparation logs
and partial caches. Audio normalization and LibriSpeech arrays are unchanged;
no T2-B1 generator has been fitted. This correction precedes the annotation
revision and any final evaluation.
