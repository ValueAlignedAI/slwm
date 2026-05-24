# T2 Latent Exploration Report

**Claim limit:** Diagnostic T2 latent exploration only: predicted/reconstructed audio and visual latent fields, retrieval probes, PCA views, and optional heuristic latent-to-media renders. No learned raw waveform/video decoder, text-generation, policy/commit, grounding, understanding, or faithful media-generation claim.
**Media renderer limit:** Optional WAV/PNG/MP4 outputs are approximate diagnostic latent-to-media renders using fixed inverse-log-mel/row-latent heuristics. The T2 model did not train a raw waveform/video decoder, phase/color/motion were discarded by extraction, and these files are not evidence of faithful media generation.

## Run

- experiment: `EXP-T2-111`
- checkpoint: `experiments/multimodal/t2/EXP-T2-111/checkpoint.pt`
- config hash: `sha256:9e7385bfd3faecb47fe0790c0a4d83ff4b1c579d26382093dab3ca0b402d7747`
- split: `validation`
- batch size: `5`
- device: `cuda`
- dtype: `torch.bfloat16`
- dataset manifest hash: `sha256:c59a1bd6bb72c6aa8d7e0859a3bca94bbf9726c852ffffca57c49c0572d794cf`
- feature spec: `{'audio_feature_dim': 80, 'audio_length': 512, 'visual_feature_dim': 256, 'visual_length': 512}`
- dataset mix: `{'audio': 0.5, 'text_code': None, 'visual_video': 0.5}`
- security scope: `Loads trusted local PyTorch checkpoints only from experiments/multimodal/t2 and writes exploration artifacts only under experiments/exploration.`

## Model / compute summary

- model: `slwm-t2-124m-vggsound-available-v0`
- variant: `slwm_audio_visual_latent`
- context length: `1024`
- latent dim: `768`
- layers / heads: `13` / `12`
- estimated parameter count: `{'adapters': 259584, 'decoders': 0, 'embeddings': 789504, 'heads': 1439569, 'policy': 0, 'processor': 123603456, 'total': 126092113}`
- training compute summary: `{'configured_steps': 1000, 'configured_batch_size': 256, 'configured_precision': 'bf16', 'accelerator_memory_mb': 21474.9140625, 'max_memory_mb': 2572.61328125}`
- dataset split counts: `{'test': {'audio_frames': 3072, 'samples': 6, 'visual_frames': 1536}, 'train': {'audio_frames': 1117696, 'samples': 2183, 'visual_frames': 558848}, 'validation': {'audio_frames': 2560, 'samples': 5, 'visual_frames': 1280}}`
- diagnostic media renderer: `{'enabled': True, 'audio_renderer': 'approximate normalized log-mel -> dB percentile scaling -> Griffin-Lim', 'visual_renderer': 'grayscale row-latent [-1,1] -> uint8 frame; repeated/panel video', 'sample_rate': 16000, 'n_fft': 400, 'hop_length': 160, 'audio_n_iter': 16, 'video_fps': 2, 'video_seconds': 4.0}`

## Samples

- `vggsound-005187--dRucPKqTFc-000007` — `people slapping`
- `vggsound-005256--rcoSOKo-qM-000566` — `electric grinder grinding`
- `vggsound-005362-0D8KognM4uY-000282` — `playing hockey`
- `vggsound-005390-0KAeUAAoVnY-000143` — `firing muskets`
- `vggsound-005392-0KIC9xmmD8g-000105` — `police radio chatter`

## Mode metrics

### `masked_av`

- audio latent MSE: `0.678711`
- visual latent MSE: `0.262470`
- audio null/persistence MSE: `1.097530`
- visual null/persistence MSE: `0.283502`
- alignment loss: `0.944919`
- retrieval: `{'retrieval_r1': 1.0, 'retrieval_r5': 1.0, 'retrieval_r10': 1.0, 'mean_rank': 1.0}`
- shuffled retrieval: `{'shuffled_retrieval_r1': 0.0, 'shuffled_retrieval_r5': 1.0, 'shuffled_retrieval_r10': 1.0, 'shuffled_mean_rank': 4.2}`

- latent_npz: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/masked_av/latent_predictions.npz`
- audio_heatmap: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/masked_av/audio_latent_prediction.png`
- visual_heatmap: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/masked_av/visual_latent_prediction.png`
- mse_curve: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/masked_av/latent_mse_over_time.png`
- retrieval_matrix: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/masked_av/av_retrieval_matrix.png`
- latent_world_pca: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/masked_av/latent_world_pca.png`
- pca_summary:
  - plot_created: `True`
  - explained_variance_ratio: `[0.3052888512611389, 0.18194720149040222]`
  - renderer: `PIL`
- diagnostic_media:
  - media_claim_limit: `Optional WAV/PNG/MP4 outputs are approximate diagnostic latent-to-media renders using fixed inverse-log-mel/row-latent heuristics. The T2 model did not train a raw waveform/video decoder, phase/color/motion were discarded by extraction, and these files are not evidence of faithful media generation.`
  - input_masked_visual_frame: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/masked_av/input_masked_visual_frame.png`
  - prediction_only_visual_frame: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/masked_av/prediction_only_visual_frame.png`
  - completed_visual_frame: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/masked_av/completed_visual_frame.png`
  - target_visual_frame: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/masked_av/target_visual_frame.png`
  - diagnostic_visual_panels: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/masked_av/diagnostic_visual_panels.png`
  - diagnostic_visual_gif: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/masked_av/diagnostic_visual_sequence.gif`
  - diagnostic_visual_mp4: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/masked_av/diagnostic_visual_sequence.mp4`
  - input_masked_audio: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/masked_av/input_masked_audio.wav`
  - prediction_only_audio: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/masked_av/prediction_only_audio.wav`
  - generated_completed_audio: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/masked_av/generated_completed_audio.wav`
  - target_reference_audio: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/masked_av/target_reference_audio.wav`

### `audio_to_visual`

- audio latent MSE: `0.666634`
- visual latent MSE: `0.289709`
- audio null/persistence MSE: `1.097530`
- visual null/persistence MSE: `0.283637`
- alignment loss: `0.915134`
- retrieval: `{'retrieval_r1': 1.0, 'retrieval_r5': 1.0, 'retrieval_r10': 1.0, 'mean_rank': 1.0}`
- shuffled retrieval: `{'shuffled_retrieval_r1': 0.0, 'shuffled_retrieval_r5': 1.0, 'shuffled_retrieval_r10': 1.0, 'shuffled_mean_rank': 4.0}`

- latent_npz: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/audio_to_visual/latent_predictions.npz`
- audio_heatmap: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/audio_to_visual/audio_latent_prediction.png`
- visual_heatmap: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/audio_to_visual/visual_latent_prediction.png`
- mse_curve: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/audio_to_visual/latent_mse_over_time.png`
- retrieval_matrix: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/audio_to_visual/av_retrieval_matrix.png`
- latent_world_pca: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/audio_to_visual/latent_world_pca.png`
- pca_summary:
  - plot_created: `True`
  - explained_variance_ratio: `[0.3735375702381134, 0.2745884954929352]`
  - renderer: `PIL`
- diagnostic_media:
  - media_claim_limit: `Optional WAV/PNG/MP4 outputs are approximate diagnostic latent-to-media renders using fixed inverse-log-mel/row-latent heuristics. The T2 model did not train a raw waveform/video decoder, phase/color/motion were discarded by extraction, and these files are not evidence of faithful media generation.`
  - input_masked_visual_frame: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/audio_to_visual/input_masked_visual_frame.png`
  - prediction_only_visual_frame: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/audio_to_visual/prediction_only_visual_frame.png`
  - completed_visual_frame: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/audio_to_visual/completed_visual_frame.png`
  - target_visual_frame: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/audio_to_visual/target_visual_frame.png`
  - diagnostic_visual_panels: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/audio_to_visual/diagnostic_visual_panels.png`
  - diagnostic_visual_gif: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/audio_to_visual/diagnostic_visual_sequence.gif`
  - diagnostic_visual_mp4: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/audio_to_visual/diagnostic_visual_sequence.mp4`
  - input_masked_audio: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/audio_to_visual/input_masked_audio.wav`
  - prediction_only_audio: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/audio_to_visual/prediction_only_audio.wav`
  - generated_completed_audio: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/audio_to_visual/generated_completed_audio.wav`
  - target_reference_audio: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/audio_to_visual/target_reference_audio.wav`

### `visual_to_audio`

- audio latent MSE: `1.051962`
- visual latent MSE: `0.275344`
- audio null/persistence MSE: `1.051556`
- visual null/persistence MSE: `0.283502`
- alignment loss: `1.319487`
- retrieval: `{'retrieval_r1': 1.0, 'retrieval_r5': 1.0, 'retrieval_r10': 1.0, 'mean_rank': 1.0}`
- shuffled retrieval: `{'shuffled_retrieval_r1': 0.0, 'shuffled_retrieval_r5': 1.0, 'shuffled_retrieval_r10': 1.0, 'shuffled_mean_rank': 3.4}`

- latent_npz: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/visual_to_audio/latent_predictions.npz`
- audio_heatmap: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/visual_to_audio/audio_latent_prediction.png`
- visual_heatmap: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/visual_to_audio/visual_latent_prediction.png`
- mse_curve: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/visual_to_audio/latent_mse_over_time.png`
- retrieval_matrix: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/visual_to_audio/av_retrieval_matrix.png`
- latent_world_pca: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/visual_to_audio/latent_world_pca.png`
- pca_summary:
  - plot_created: `True`
  - explained_variance_ratio: `[0.35011547803878784, 0.1794305145740509]`
  - renderer: `PIL`
- diagnostic_media:
  - media_claim_limit: `Optional WAV/PNG/MP4 outputs are approximate diagnostic latent-to-media renders using fixed inverse-log-mel/row-latent heuristics. The T2 model did not train a raw waveform/video decoder, phase/color/motion were discarded by extraction, and these files are not evidence of faithful media generation.`
  - input_masked_visual_frame: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/visual_to_audio/input_masked_visual_frame.png`
  - prediction_only_visual_frame: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/visual_to_audio/prediction_only_visual_frame.png`
  - completed_visual_frame: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/visual_to_audio/completed_visual_frame.png`
  - target_visual_frame: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/visual_to_audio/target_visual_frame.png`
  - diagnostic_visual_panels: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/visual_to_audio/diagnostic_visual_panels.png`
  - diagnostic_visual_gif: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/visual_to_audio/diagnostic_visual_sequence.gif`
  - diagnostic_visual_mp4: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/visual_to_audio/diagnostic_visual_sequence.mp4`
  - input_masked_audio: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/visual_to_audio/input_masked_audio.wav`
  - prediction_only_audio: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/visual_to_audio/prediction_only_audio.wav`
  - generated_completed_audio: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/visual_to_audio/generated_completed_audio.wav`
  - target_reference_audio: `experiments/exploration/t2_latent_generation/EXP-T2-111_validation_step0000_20260523T152538Z/visual_to_audio/target_reference_audio.wav`

## Interpretation limits

- These are latent-field diagnostics, not learned/raw-decoder audio/video generation.
- Heatmaps visualize model-predicted latent tensors and errors.
- Cross-modal modes are diagnostic/off-distribution probes unless evaluated separately.
- Tiny VGGSound validation/test splits prevent strong natural-AV conclusions.
- Optional WAV/PNG/MP4 files are heuristic renders of predicted latents, not learned raw-media decodes.
