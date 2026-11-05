# T2-B1 evaluation details registered before final scores

All world checkpoints and P1-S1 actors/thresholds must be frozen before opening
final test signals for scoring. Preserve every result, including failed cases.
Training source snapshots remain unchanged. The final test metadata was used
only for official identity separation, clock/support and annotation integrity.

Speech: all48 clips from each fresh Libri set,160ms and320ms erasures. Place one
gap at the midpoint of the longest whole word in the central60% of the clip
(ties by time); if there is no such word, use the temporal midpoint. Gap placement
uses annotations for the benchmark mask only, never as a model input. All models
receive identical physical erasure before Fourier analysis; inputs whose FFT
support intersects a hidden sample are masked. Score normalized true-gap mel
MAE, whole-clip WER and raw recognizer text. Clamp every known PCM sample exactly.
Use the same10ms internal boundary blend for all generated/interpolated repairs.
Controls: recorded audio, silence, linear-waveform interpolation, fine-frame
observed-only mel interpolation plus Vocos, and oracle mel plus Vocos. The oracle
control measures the external renderer ceiling and is never a practical input.

Whisper-tiny.en revision87c7102498dcde7456f24cfd30239ca606ed9063 is frozen and
external. Normalize both reference and hypothesis with its standard English
normalizer before primary WER; retain unnormalized errors. Short clips allow
256 new tokens. Long outputs are decoded in consecutive nonoverlapping20s
windows, including the final remainder; store source intervals and all raw
transcripts so no30s truncation can be mistaken for whole-output scoring.
Bootstrap paired effects by person with10,000 seeded resamples; report seed
effects separately. The original5% relative WER gate remains unchanged.

Shared fields:16 deterministic nonoverlapping128-packet windows per AMI test
person, starting at40s and advancing8s; retain only windows with at least four
wholly contained official words. Encode audio, video and explicitly supplied
text independently with zero appearance cue. Retrieve within person among all
eligible windows, reporting actual chance1/N. Compare same-person wrong-time
pairings, all architecture/control arms, and field phone/CTC outputs. Report
future8-packet error beyond the2-packet guard against last-observation persistence.
These are representation diagnostics, not motor-world or semantic understanding.

Long generation: every fixed40.96s interval at80s, all12 AMI test people, main
SLWM/Transformer arms in both seeds. Retain input64×64 video and original sound,
audio→video (original soundtrack explicitly labeled), video→audio (no source
audio decoded), model CTC text, and text→audio/video for both fixed long prompts.
Provide a static training appearance for generated video; native resolution64.
Run reset-memory and eight-chunk-prefix controls for main seed91 arms, plus
same-person wrong-time audio/video conditioning and static/mean/silent controls.
Report all-frame code error, motion variation, unique frames, frame/sample count,
coverage of the complete stream, transcript error, latency and memory drift.
Constant appearances do not count as generated scene knowledge. No pretrained
image/video generator is used. Audio renderer weights are external and fixed.

Frozen P1-S1 decisions are evaluated only after all policy fitting and commitment
calibration. Report each person's cost against all registered fixed strategies,
both architectures and seeds. Commit coverage never removes ticks from utility.
Paced replay is a continuous operational file loop; sensing does not alter the
physical world. A later physical-action model needs intervention data separately.
