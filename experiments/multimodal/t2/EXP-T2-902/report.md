# Sprint T2 Audio/Visual Latent Report — EXP-T2-902

- Model variant: `slwm_audio_visual_latent`
- Parameter count: `161233`
- Prepared corpus: `artifacts/t2_audio_visual/external_t2_control_v0`
- Physical/effective batch size: `4` / `4`
- Eval batch size: `4`
- Initial → validation total loss: `1.3584610223770142` → `1.3338103294372559`
- Initial → audio latent MSE: `0.9341449737548828` → `0.9354394674301147`
- Initial → visual/video latent MSE: `0.28926411271095276` → `0.2686299979686737`
- Audio-video retrieval R@1/R@5: `0.5` / `1.0`
- Shuffled retrieval R@1: `0.25`
- Prediction loss decreased: `False`
  - Audio decreased: `False`; visual decreased: `True`; total decreased: `True`
- Checkpoint: `experiments/multimodal/t2/EXP-T2-902/checkpoint.pt`

## Scope
Audio and visual/video latent prediction only. No text generation, raw media generation, policy, or hallucination claim is made.

## Claim limits
Only T2 latent prediction, audio spectral proxy, visual latent error, AV retrieval/correspondence, shuffled/null controls, throughput, and memory metrics may be reported.
