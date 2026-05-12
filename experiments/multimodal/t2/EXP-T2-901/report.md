# Sprint T2 Audio/Visual Latent Report — EXP-T2-901

- Model variant: `slwm_audio_visual_latent`
- Parameter count: `4426`
- Prepared corpus: `artifacts/t2_audio_visual/generated_av_smoke_v0`
- Initial → validation total loss: `0.7444257140159607` → `0.7526866793632507`
- Initial → audio latent MSE: `0.28608280420303345` → `0.26910488307476044`
- Initial → visual/video latent MSE: `0.31955766677856445` → `0.3449752628803253`
- Audio-video retrieval R@1/R@5: `0.25` / `1.0`
- Shuffled retrieval R@1: `0.25`
- Prediction loss decreased: `False`
  - Audio decreased: `True`; visual decreased: `False`; total decreased: `False`
- Checkpoint: `experiments/multimodal/t2/EXP-T2-901/checkpoint.pt`

## Scope
Audio and visual/video latent prediction only. No text generation, raw media generation, policy, or hallucination claim is made.

## Claim limits
Only T2 latent prediction, audio spectral proxy, visual latent error, AV retrieval/correspondence, shuffled/null controls, throughput, and memory metrics may be reported.
