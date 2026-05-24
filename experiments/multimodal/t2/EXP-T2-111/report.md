# Sprint T2 Audio/Visual Latent Report — EXP-T2-111

- Model variant: `slwm_audio_visual_latent`
- Parameter count: `126092113`
- Prepared corpus: `artifacts/t2_audio_visual/vggsound_5k_available_v0`
- Physical/effective batch size: `256` / `256`
- Eval batch size: `5`
- Initial → validation total loss: `2.1459238529205322` → `1.0551384687423706`
- Initial → audio latent MSE: `1.3828125` → `0.70703125`
- Initial → visual/video latent MSE: `0.59765625` → `0.251953125`
- Audio-video retrieval R@1/R@5: `1.0` / `1.0`
- Shuffled retrieval R@1: `0.0`
- Prediction loss decreased: `True`
  - Audio decreased: `True`; visual decreased: `True`; total decreased: `True`
- Checkpoint: `experiments/multimodal/t2/EXP-T2-111/checkpoint.pt`

## Scope
Audio and visual/video latent prediction only. No text generation, raw media generation, policy, or hallucination claim is made.

## Claim limits
Only T2 latent prediction, audio spectral proxy, visual latent error, AV retrieval/correspondence, shuffled/null controls, throughput, and memory metrics may be reported.
