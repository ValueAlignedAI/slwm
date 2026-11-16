# Study X2 — Wave 8A-r2 (`runs_wave8a_r2`) Pre-Registration Brief: Across-Depth `Z_mem ⊕ Z_task` Isolation, Audio-Only Goal Routing, & Calibrated Epistemic Commitment

## 1. Scientific Motivation & Remediated Root Causes from Wave 8A v1

Wave 8A v1 (`runs_wave8a`, `1,800` steps) established that `Z_mem ⊕ Z_task` head/readout partitioning improved `T = 1024` dynamic-visual associative overwrite recall over Wave 7 (`6.3%` chance -> `58.3% ± 26.0%`), achieved `99.68%` unseen sensor-corruption detection AUROC, and avoided `SlidingWindow-W64-GQA`'s `0.0%` post-eviction collapse (`37.5% ± 10.2%` paired causal switch). However, independent Gate 4 peer review rejected Wave 8A v1 on four falsifiable findings:
1. **Across-Depth Trunk Task Leakage & Insufficient Associative Exposure (`dyn_assoc_T1024 = 58.3% ± 26.0% < 95.0%`, `T160_paired_causal_switch = 37.5%` vs `43.2%` `Spatial-Only`):** `_spatial_cross_modal_attn_wave8a` allowed `0.22x` residual task-gradient flow into `SLOT_TEXT` across all 24 trunk layers (`L0..L23`), while associative episodes never co-occurred with goal-query prefix tokens (`{4..7}`).
2. **Multi-Step Latent Rollout Value-Spike Sensitivity (`unseen_mppi = 8.9% ± 5.9%` vs `77.1%` `no_jepa_subgoal`, `95.3%` `H = 12` Oracle ceiling):** Uncapped `12`-step learned latent rollout drift and terminal value lookups near thin-wall boundaries degraded multi-step rollout scores below single-step subgoal selection.
3. **Text-Only `H_goal` Gate Routing & `audio_only` Under-Exposure (`vault_spoken_audio_only_disc = 52.6% ± 1.6%`):** `H_goal` recurrent write gates read `SLOT_TEXT` exclusively (`0.0` when text was masked), leaving chronological two-utterance `audio_only` directives (`t = 0..15` `GRID` color speech, `t = 16..31` `SpeechCommands` direction speech) unable to latch the spoken color across the `16`-tick gap.
4. **Uncalibrated `PolicyCommitGate` Input Norms & Unsupervised Arena Discrepancy Contamination (`commit_selective_routing = 50.0% ± 0.0%`):** Raw `1152`-D trunk sums overwhelmed scalar visual-proprioceptive innovation features, while `commit_loss` supervised `target_commit = 0` on oblique `xArm6` (`scen_arr = -2`) and `PyMunk-64` (`scen_arr = -1`) episodes where top-down `vis_ee_curr` is unsupervised.

**Pre-Registered Remediations in Wave 8A-r2 (`configs/x1b/study_x2_wave8a_r2.json`, `0` Extra Parameters, `0` Extra State Bytes):**
- **Exact Across-Depth `Z_mem` Trunk Isolation (`assoc_w = 5.0`):** Restrict `0.20x` spatial cross-modal gradient into `SLOT_TEXT` strictly to `layer_idx == 0` (calibrating non-associative `text_emb` rows `96..4095`) and enforce `stop_gradient(x_txt)` across all layers `L1..L23` plus `stop_gradient(txt_slot)` on `H_goal` query/write projections; apply `50%` slot-0 `{4..7}` prefix augmentation (`t = 0..23`) and `100%` active beyond-window queries (`t = 56..63`).
- **Balanced `H = 6` Pose-Anchored Rollout Functional:** Anchor per-step rollout object pose with `0.65 * raw_obj + 0.35 * vis_obj_step`, cap boundary value spikes (`min(v_grid, 3.5)`), and weight `roll_term = 0.14 * rollout_ret - 0.12 * fin_v` identically across both learned MPPI and the `Oracle-Perception x Oracle-Dynamics` ceiling (reporting both `oracle_mppi_ceiling` and `oracle_no_jepa_subgoal_ceiling`).
- **Zero-Drift Per-Tick `H_goal` Modality Routing & Causal Speech Supervision:** Route `lang_goal_src = txt_p * stop_grad(txt_slot) + aud_p * (1 - txt_p) * aud_mean` (`0.0` audio drift when text is present; `100%` pure waveform write when `txt_p == 0` & `aud_p == 1`), switch `H_goal` query to pure `q_reg_g` when `txt_p == 0`, evaluate batched `[100% audio_only, 70% joint / 30% text_only]` `run_init_disc` on every step, and mask pre-direction (`t < 28`) audio-only `target_goal_xy` supervision.
- **Arena-Calibrated `PolicyCommitGate` & Clean-Tick JEPA/SIGReg Masking:** Normalize `commit_in` trunk features (`0.08 * l2_normalize(stop_grad(reg_sum))`), supervise `commit_loss` strictly on calibrated arena episodes (`scen_arr >= 0`) with `50%` `180°` flip / `50%` in-domain camera mount extrinsic translation (`±6..12` px shift, strictly disjoint from all 4 unseen evaluation corruptions) for `Mode 1` and RGB static + `ee_only` jump for `Mode 3`, and mask `jepa_1/2/4`, `unc_loss`, `sig_v`, `av_nce`, and `recon_v_loss` by `clean_tick_mask = (commit_target == 0) * valid`.

---

## 2. Controlled Cohort Matrix (`453M` Scale, `n = 3` Seeds `{2001, 2002, 2003}` = 15 Runs)

All five cohorts execute in `runs_wave8a_r2` on `NVIDIA A100-SXM4-40GB` accelerators (`bfloat16` mixed precision with `float32` recurrent/norm states, preserving `runs_wave8a` and `study_x2_wave8a.json` untouched) under identical sequence-tensor budgets (`2,200` steps, batch size `8`, sequence length `T = 64`, `1,126,400` sequence-ticks per run), identical frozen `v2` datasets (`wave8a_dataset_cards.json`), strict `ee_only` proprioception, `< 0.01%` strict parameter parity, and identical `Core / Strict System` `B = 1` streaming state footprints:

| Cohort ID | Architecture | Strict Params | Core / Strict System `B = 1` State (`MiB`) | Compiled Latency (`ms/tick`) | `zmem_isolation` | `enable_assoc_loss` | Controlled Experimental Role |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `w8a_slwm_x2_isolated_450m` | `slwm_x1b` (`3:1` GatedDelta + `W=8` local attn) | `453,199,938` | `8.7898 / 8.8533 MiB` | `5.08 ± 0.04` | `True` | `True` | Primary `SLWM-X2` joint model (`assoc_w = 5.0`) |
| `w8a_ablation_no_zmem_iso_450m` | `slwm_x1b` (`3:1` GatedDelta + `W=8` local attn) | `453,199,938` | `8.7898 / 8.8533 MiB` | `4.57 ± 0.02` | `False` | `True` | Shared-trunk ablation (`zmem_isolation = False`, `assoc_w = 5.0`) |
| `w8a_ablation_spatial_control_only_450m` | `slwm_x1b` (`3:1` GatedDelta + `W=8` local attn) | `453,199,938` | `8.7898 / 8.8533 MiB` | `5.11 ± 0.06` | `True` | `False` | Spatial-control-only ceiling anchor (`assoc_w = 0.0`) |
| `w8a_sliding_window_w64_gqa_450m` | `sliding_window_gqa_transformer` (`9:1` GQA, `W=64`) | `453,224,778` (`+0.0055%`) | `13.5264 / 13.5898 MiB` | `4.53 ± 0.06` | `True` | `True` | Parameter-matched GQA bounded-KV Transformer (`assoc_w = 5.0`) |
| `w8a_diagonal_ssm_450m` | `diagonal_ssm_hybrid` (`3:1` DiagonalSSM + `W=8` local attn) | `453,199,938` | `4.9138 / 4.9773 MiB` | `4.48 ± 0.03` | `True` | `True` | Parameter-matched vector-state recurrence (`assoc_w = 5.0`) |

---

## 3. Pre-Registered Fail-Closed Gate 4 Criteria (`evaluate_gate4_checks`)

In addition to all Wave 8A v1 recall, scene-grounding, `9`-metric `<= 2.0 pp` non-inferiority, `15`-cell $K \times T$ capacity, `58` hierarchical `(Seed x World/Scene)` CIs, and median-seed `dev` showcase requirements (`inference_showcase/wave8a_r2/dev/`), `study_x2_wave8a_r2.json` enforces four explicit absolute capability floors on `w8a_slwm_x2_isolated_450m`:
1. **Multi-Step MPPI Capability & Rollout Non-Degradation:** `unseen_mppi >= 65.0%` **and** `unseen_mppi >= no_jepa_subgoal - 2.0 pp`.
2. **Calibrated Epistemic Routing:** `commit_selective_routing >= 80.0%` across clean vs. unseen sensor-fault streams.
3. **Pure Spoken Audio-Only Physical `xArm6` Grounding:** `vault_spoken_audio_only_disc >= 70.0%` opposite-direction discrimination on untouched speaker- and scene-disjoint `vault` episodes, alongside `>= +10.0 pp` (`lo > 0`) scene-grounding lift over both `mu_train[color_id]` and contrast-preserving disjoint-scene shuffle, plus `>= 20.0%` (`lo > 0`) paired `180°`-flip dual-pair localization accuracy.
4. **Long-Horizon Recall & Causal Goal-Switch Floors:** `dyn_visual_assoc_T1024 >= 95.0%` and `T160_paired_causal_switch >= 50.0%` (within `<= 2.0 pp` of `w8a_ablation_spatial_control_only_450m`).
