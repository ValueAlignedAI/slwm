# Sprint T2 Audio/Visual Latent Report — EXP-T2-903

- Model variant: `slwm_audio_visual_latent`
- Parameter count: `161233`
- Prepared corpus: `artifacts/t2_audio_visual/vggsound_5k_available_v0`
- Physical/effective batch size: `4` / `4`
- Eval batch size: `4`
- Initial → validation total loss: `1.557301640510559` → `1.398801565170288`
- Initial → audio latent MSE: `1.1178911924362183` → `0.982588529586792`
- Initial → visual/video latent MSE: `0.29917585849761963` → `0.2779894471168518`
- Audio-video retrieval R@1/R@5: `0.25` / `1.0`
- Shuffled retrieval R@1: `0.25`
- Prediction loss decreased: `True`
  - Audio decreased: `True`; visual decreased: `True`; total decreased: `True`
- Checkpoint: `None`

## Scope
Audio and visual/video latent prediction only. No text generation, raw media generation, policy, or hallucination claim is made.

## Claim limits
Only T2 latent prediction, audio spectral proxy, visual latent error, AV retrieval/correspondence, shuffled/null controls, throughput, and memory metrics may be reported.
