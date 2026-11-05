# Broad shared fields, long generation and continuous sensing

This run has three distinct artifacts:24M audio/video/phonetic-text world fields,
124/125M English language models, and small frozen-world sensing actors. The
world text input uses a fixed CMU pronunciation dictionary and training-estimated
word timing. It is not a general semantic English encoder. Native generated
video is64×64 gray motion over a declared supplied training appearance.

The selected world/actor exports are under
`experiments/multimodal/t2/broad-stream-20261105/inference_bundle/`; the selected
language exports are under
`experiments/text/t1/extended-throughput-20261105/inference_exports/`.
Full best/resume/final checkpoints, optimizers, RNG states, curves, sources and
all media remain in the original experiment directories on Fedora and the local
artifact mirror. Large binaries are deliberately outside Git. Check the delivery
manifests rather than assuming a successful copy.

Install the repository's `research-stream` dependencies plus matching-platform
PyTorch/Torchaudio, Transformers/tiktoken for the language/evaluation tools, and
FFmpeg/FFprobe with a CPU H.264 encoder (`libx264` or `libopenh264`). The measured
Fedora environment is recorded with each experiment. The portable world bundle
includes the separately counted frozen13,531,650-parameter Vocos waveform
renderer and its original provenance. It supplies waveform rendering, not the
learned phonetic or video predictions. No pretrained image/video generator is used.

Run these from the repository root after exports complete, using the appropriate
Python environment. `WORLD`, `CODEC`, `ACTOR`, `AUDIO` and `VIDEO` below stand for
ordinary filesystem paths; inputs may be long recordings. Defaults process40.96s
for generation and31 received chunks for the continuous actor.

```bash
python -m exploration.broad_stream_infer --checkpoint WORLD --codec-dir CODEC \
  --mode audio-to-video --audio AUDIO --seconds 40.96 --output OUTPUT
python -m exploration.broad_stream_infer --checkpoint WORLD --codec-dir CODEC \
  --mode video-to-audio --video VIDEO --seconds 40.96 --output OUTPUT
python -m exploration.broad_stream_infer --checkpoint WORLD --codec-dir CODEC \
  --mode text-to-av --text "A sufficiently long passage..." \
  --min-text-seconds 40.96 --output OUTPUT
python -m exploration.broad_stream_infer --checkpoint WORLD --codec-dir CODEC \
  --mode repair --audio AUDIO --gaps 10.0:10.16 --seconds 40.96 --output OUTPUT
python -m exploration.stream_sensing_infer --world WORLD --actor ACTOR \
  --codec-dir CODEC --audio AUDIO --video VIDEO --goal both --pace --output OUTPUT
```

Audio→video retains the original audio soundtrack and generates only motion.
Video→audio never decodes source audio in the generation CLI. Text→audio/video
receives no reference timings; the optional minimum duration globally stretches
the training-estimated word times, never repeats a short generated clip.
Repair erases hidden PCM before analysis and preserves every observed PCM sample
exactly. Prefix-only and reset-memory flags retain the same code path and expose
free-state collapse or dependence on recurrence. All proposals, state traces,
actual model CTC text, PCM sound and verified-frame-count videos are saved.
Add `--video VIDEO` to repair mode to supply synchronized observed video; the
first observed video packet provides an explicit static appearance cue.

Long-media rendering follows the continuous field rollout; it is not a live
camera/audio transport. The model receives1.365s chunks plus the declared FFT
analysis support. The Vocos waveform renderer uses an additional48 future
predicted mel frames (512ms) of context. Model-only decision timing and waveform
delivery timing are different; the sensing actor does not invoke the renderer.

The sensing CLI makes30 automatic decisions after one bootstrap chunk, with
persistent working memory and no future labels or transcripts. It emits
listen/look/both/wait, commitment decisions and future proposals without another
user request. With `--pace`,30 decisions occupy40.96s after bootstrap; it logs
latency/deadlines and verifies both world and actor weights remain unchanged.
This is a recorded-file operational loop with hypothetical sensor-access costs.
Features are prepared before replay. Paced inference chooses a sensor before
waiting for the next chunk's recorded arrival; it processes that chunk only
after the deadline, logging decision/read/proposal times separately.
Actions control observations, not physical
motion; motor-world transition/reward data and a faster loop are still needed.

The main B1 training source/config/protocol snapshot is immutable. Person-disjoint
selection chooses world checkpoints; six world-training people are independently
held out from risk-head fitting, four completely new people calibrate commitment,
and12 different people provide final operational tests. Every uncommitted tick
still counts in cost. Test scores never select weights or thresholds. Reused
historical GRID/CREMA results remain separate from this fresh broader evaluation.

Read the experiment reports, all-case `watch.html`, representation/recognizer
JSONs and P1 cost/coverage table together. Lower spectral error, nonconstant
pixels, CTC/readout alignment or a continuously running loop alone do not prove
intelligible generation, a general world model or a SLWM-specific advantage.

For language continuation, use the separate export with
`python -m exploration.scale_text_infer --checkpoint CHECKPOINT --prompt TEXT`.
The output records actual checkpoint token exposure.512-token context and severe
undertraining remain; the model is not an instruction-following factual assistant.
