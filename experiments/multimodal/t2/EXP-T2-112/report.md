# Sprint T2 Audio/Visual Latent Report — EXP-T2-112

- Model variant: `slwm_audio_visual_latent_no_spectral`
- Parameter count: `125762641`
- Prepared corpus: `artifacts/t2_audio_visual/vggsound_5k_available_v0`
- Physical/effective batch size: `256` / `256`
- Eval batch size: `5`
- Initial → validation total loss: `2.020362138748169` → `0.9785449504852295`
- Initial → audio latent MSE: `1.2578125` → `0.65625`
- Initial → visual/video latent MSE: `0.60546875` → `0.224609375`
- Audio-video retrieval R@1/R@5: `1.0` / `1.0`
- Shuffled retrieval R@1: `0.0`
- Prediction loss decreased: `True`
  - Audio decreased: `True`; visual decreased: `True`; total decreased: `True`
- Checkpoint: `experiments/multimodal/t2/EXP-T2-112/checkpoint.pt`

## Scope
Audio and visual/video latent prediction only. No text generation, raw media generation, policy, or hallucination claim is made.

## Claim limits
Only T2 latent prediction, audio spectral proxy, visual latent error, AV retrieval/correspondence, shuffled/null controls, throughput, and memory metrics may be reported.
