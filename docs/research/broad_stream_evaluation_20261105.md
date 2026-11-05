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

Pre-test secondary decoder diagnostic: for the four main world/seed arms, also
render `_conserved_context` with every physically observed fine mel frame
restored at renderer input. Keep each hidden model mel proposal exactly the
same. Use the same erased-waveform FFT visibility and the same frozen Vocos,
fade and known-PCM clamp; no original hidden sample, target feature, phase or
extra fitted weight enters this variant. Preserve original outputs and primary
gates. Save the fine mask and both mel arrays; compare paired WER against the
original readout and fine interpolation, both gaps/sets/seeds. These secondary
conditions test decoder context conservation, not an improved hidden prediction
or a retroactive primary pass. Register this before any B1/P1 final test score.

Whisper-tiny.en revision87c7102498dcde7456f24cfd30239ca606ed9063 is frozen and
external. Normalize both reference and hypothesis with its standard English
normalizer before primary WER; retain unnormalized errors. Short clips allow
256 new tokens. Long outputs are decoded in consecutive nonoverlapping20s
windows, including the final remainder; store source intervals and all raw
transcripts so no30s truncation can be mistaken for whole-output scoring.
Mean whole-clip WER is primary, with corpus word-weighted WER reported separately.
Bootstrap paired effects by person with10,000 seeded resamples; report seed
effects separately. The original5% relative WER gate remains unchanged.

Shared fields:16 deterministic nonoverlapping128-packet windows per AMI test
person, starting at40s and advancing8s; retain only windows with at least four
wholly contained official words. Encode audio, video and explicitly supplied
text independently with zero appearance cue. Retrieve within person among all
eligible windows, reporting actual chance1/N. Compare same-person wrong-time
pairings, all architecture/control arms, and field phone/CTC outputs. Report
future8-packet error beyond the2-packet guard against last-observation persistence
and the normalized-zero training-mean prior. A persistence win alone cannot rule
out a generic average forecast.
These are representation diagnostics, not motor-world or semantic understanding.
Score actual learned field CTC strings separately from the external recognizer's
transcripts of generated sound; both use the same fixed English normalization.
Provided-phone text cases are labeled, so they cannot masquerade as audio/video
recognition or unconstrained language generation.

Secondary AV repair: use the same40.96s/80s interval for all12 AMI test people
and all four main architecture/seed arms. Place one160ms physical gap by the same
word/midpoint rule as LibriSpeech. Compare the exact same learned model with
audio only, aligned video, static first-packet video, same-person video80s later,
and ±240ms video. Exact240ms shifts use linear interpolation of the fixed PCA
codes; all video conditions share the same observed first-packet appearance cue.
Keep quiet/no-reference cases explicitly. Known PCM is exact. This is a secondary
transfer probe, not a new primary speech gate or a fitted timing-rejection policy.

Long generation: every fixed40.96s interval at80s, all12 AMI test people, main
SLWM/Transformer arms in both seeds. Retain input64×64 video and original sound,
audio→video (original soundtrack explicitly labeled), video→audio (no source
audio supplied to the model), model CTC text, and text→audio/video for both fixed long prompts.
Provide a static training appearance for generated video; native resolution64.
Run reset-memory and eight-chunk-prefix controls for main seed91 arms, plus
same-person wrong-time audio/video conditioning and static/mean/silent controls.
Report all-frame code error, motion variation, unique frames, frame/sample count,
coverage of the complete stream, transcript error, latency and memory drift.
Constant appearances do not count as generated scene knowledge. No pretrained
image/video generator is used. Audio renderer weights are external and fixed.

Horizon diagnostic, specified before test access: compare all saved normal,
reset, prefix and wrong-time proposals in the same packet bins after the
eight-chunk boundary:0–128,128–256,256–512 and512–704 packets beyond the prefix.
This is0–5.461,5.461–10.923,10.923–21.845 and21.845–30.037s after observations
stop in the prefix arm. Normal/reset arms continue receiving observations.
Report mel MAE, code/motion MSE, training-mean and static-motion errors with
person-cluster intervals. Use persisted float16 proposals and disclose that
quantization; original per-case metrics remain full precision. No new primary
gate or weight selection follows from this diagnostic. Prefix mode evolves
persistent state with absent sensors; it does not feed predicted media back as
observations or demonstrate action-conditioned planning.

Frozen P1-S1 decisions are evaluated only after all policy fitting and commitment
calibration. Report each person's cost against all registered fixed strategies,
both architectures and seeds. Commit coverage never removes ticks from utility.
Paced replay is a continuous operational file loop; sensing does not alter the
physical world. A later physical-action model needs intervention data separately.
