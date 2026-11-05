# GRID source-quality review before model training

The original input rule requires at least50 physical video packets so1.28s
contexts and real shifted controls are valid. The0.5% global exclusion alarm
stopped preparation for review. No speech weights or thresholds had been fitted.

The review identifies90 CRC-correct members of the official speaker08 normal
video ZIP with only2–11 readable video frames. Every flagged file was checked
with `ffprobe -count_frames`, its raw SHA256 and official ZIP-member CRC. These
are genuinely short video streams alongside normal-length audio. They occur
only in training-person08:88 training clips and2 training-person composition
diagnostics. No validation, calibration or final-test clip is affected.

Keep the original50-packet criterion and all speaker roles. Record these exact
files as exclusions; their current metadata SHA256 is
`edfbe4cb1054278013f8a6456d82b1c4abaed195be33a56fcb75187526558e93`.
The source ZIP acquisition SHA256 is
`80bb8325742b04055ff0f7205c6eded60d96aa0d9205cb8ae24de6e833808757`.
Only this reviewed exclusion list may pass the alarm. Any changed list still
stops preparation. The previous failed preparation log and complete90-file
review remain under the content-sync artifact directory. This is an input-only
source review, with no loss-based filtering, crop tuning or test-person removal.
