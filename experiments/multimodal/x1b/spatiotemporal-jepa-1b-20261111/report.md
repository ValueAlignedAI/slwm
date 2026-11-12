# Study X1B — Spatiotemporal Latent-JEPA, Zero-Lag Multi-Rate GatedDelta & Innovation-Gated MPC at 1.008B Scale

- **Sprint / Study ID:** `X1B-20261111` (`experiments/multimodal/x1b/spatiotemporal-jepa-1b-20261111`)
- **Branch:** `research/x1b-spatiotemporal-jepa-mpc-20261111` (strictly isolated from concurrent G11 work, local Fedora GPU, and `~/.local/state/slwm/`)
- **Accounting Standard:** `strict_total_parameters` (`AGENTS.md` §4 & §10; `< 0.002%` parity across all `1.008B` models and `< 0.006%` parity across all `152M` factorial models)
- **Compute Ledger:** `124.40 / 500.00` Colab compute units expended across **84 total cloud runs** (`Wave 1` diagnostic cohort + `Wave 2` scaling diagnostic cohort + `Wave 3` final Zero-Lag Multi-Rate cohort); initial balance `965.17` → final balance `840.77`; `0` active Colab sessions (`current_rate_per_hr = 0.0`). All `1.008B` and `453M` runs executed on **Google TPU v6e-1 (`33.55 GB` HBM)**; `152M` 3-seed factorials executed in parallel on NVIDIA `A100-SXM4-40GB`.
- **Dual-HDD Artifact Custody (`192.168.1.200`, Fedora):** `28 / 28` Wave 3 runs (plus `28 / 28` Wave 2 and `28 / 28` Wave 1 runs preserved under `runs_wave2/` and `runs_wave1/`) verified with `MIRROR_VERIFIED_2_COPIES` across independent physical disks `fbbb53d9-dd42-4f19-95e6-dadab02a201e` (`/mnt/slwm-checkpoints-a`) and `c961a363-8d77-4fba-a48d-622b00a6d8c3` (`/mnt/slwm-checkpoints-b`).

---

## 1. Hypotheses & Pre-Registered Remedies for G0–G10 Bottlenecks

Prior studies (`G0`–`G10`) plateaued on physical forecasting, spatial precision, long-horizon working memory, closed-loop planning, and adversarial camera spoofing due to five structural bottlenecks. Study X1B tested five pre-registered architectural remedies at `152M` (3 seeds × 8 variants = 24 runs), `453M` (`SLWM-X450M`), and `1.008B` (`SLWM-X1B` vs matched `DenseTransformer-X1B` and `LegacyScalarHybrid-X1B`):

1. **H1 — Spatiotemporal Field Preservation (`S = 24` Structured Slots vs G5–G10 1-Vector Bottleneck):** Preserving a `64×64` overlapping Conv2D + `4×4` spatial patch grid (`16` visual slots) alongside `2` spectral/onset audio slots, `1` proprioceptive slot, `1` role-structured `4096`-vocab text slot, `1` motor-action slot, and `3` persistent working-memory registers (`S = 24` slots/tick) resolves `1-px` thin-barrier blindness (`AUROC = 1.000` vs `0.583` under `16×16` G8 subsampling) and eliminates cross-modal representational collapse (`90.6%` real audio-visual `R@1` at `1.008B` vs `1.56%` random chance on `LegacyScalarHybrid-1B`).
2. **H2 — Position-Free Householder Gated-Delta Matrix Memory + Directed Cross-Modal Grounding:** Replacing G5/G10 scalar exponential retention and RoPE-only temporal attention with a multi-head Householder `GatedDelta` matrix recurrence ($M_t^{(h)} = \alpha_t^{(h)} M_{t-1}^{(h)} + \beta_t^{(h)}(v_t^{(h)} - \alpha_t^{(h)} M_{t-1}^{(h)} k_t^{(h)}) (k_t^{(h)})^\top$) paired with symbol-pure key routing enables **100.0% selective key-value overwrite recall at 16× out-of-distribution horizon extrapolation (`T=32` train → `T=512` eval)**, where `LegacyScalarHybrid-1B` scores `6.25%` (chance) and `DenseTransformer-1B` collapses to `6.25%` at `T=256` and `37.5%` at `T=512`.
3. **H3 — Zero-Lag Multi-Rate Lead-Lag Kinematics at Deep Scale (`L = 23`):** Splitting temporal slot filtering into a zero-lag lead-lag bypass (`0.82 u_t + 0.18 c_t`, $\tau \in [0.35, 10]$ ticks) on sensory slots (`0..20`) and multi-timescale retention (`0.20 u_t + 0.80 c_t + \text{assoc\_out}`, $\tau \in [1.2, 128]$ ticks) on `SLOTS_REGISTER` (`21..23`) eliminates multi-layer IIR phase lag at `L = 23`, lifting `1.008B` closed-loop MPC from `35.4%` (Wave 2 pure-IIR recurrence) to **`77.1%`** (`+41.7 pp`), surpassing both `DenseTransformer-1B` (`70.8%`) and `LegacyScalarHybrid-1B` (`68.8%`).
4. **H4 — Action-Conditioned Residual-EMA-JEPA + SIGReg Anti-Collapse + Receding-Horizon Latent-CEM-MPC:** Predicting unit-scale physical Euler residuals in EMA-target latent space with variance-covariance `SIGReg` regularization prevents dimensional collapse (`Participation Ratio = 10.7%` at `1.008B` vs `1.1%` without `SIGReg` and `-9.9%` forecast skill without residual integration) and enables 6-step Latent-CEM/MPPI-MPC (`77.1%` at `1.008B` vs `41.7%` for G8 3-candidate random shooting, `+35.4 pp`).
5. **H5 — Innovation-Driven `PolicyCommitGate` Against G6 Camera Spoofing & Unsupported Claims:** Conditioning commitment mode selection (`commit_full`, `commit_trusted_proprio`, `probe_or_abstain`, `wait`) on normalized latent innovation surprise, cross-modal visual-proprioceptive pose discrepancy, and modality-presence masks reduces 180° inverted-camera spoofing regret (`0.306` vs `0.404` `always_commit` and `0.370` on `DenseTransformer-1B`) and eliminates unsupported claims under missing sensory grounding (`0.0%` vs `100.0%` `always_commit`) while maintaining `67.1%` clean-task usefulness.

---

## 2. Iterative Scientific Discovery Arc (Wave 1 → Wave 2 → Wave 3)

All three 28-run experimental waves (`84` runs total) are preserved transparently in the repository and on Fedora's dual HDDs:

1. **Wave 1 (`wave1_diagnostic_summary_metrics.json`, `runs_wave1/`):** Initial 28-run diagnostic cohort exposed two planner/head bottlenecks: (a) symmetric CEM mean-averaging across bimodal `+y / -y` obstacle-avoidance trajectories drove actions straight into central `1-px` barriers (`0%` thin-barrier clearance), and (b) `rms_norm` right before the linear proprioceptive head erased radial distance scale (`-2.4%` skill vs persistence at `1B`). Resolved via sharp MPPI Boltzmann weighting (`evals/x1b_mpc.py`), unnormalized affine `clean_prop_emb` + egocentric push kinematics (`models/x1b_jepa.py`), and causal Read-Before-Write in `GatedDelta`.
2. **Wave 2 (`wave2_scaling_diagnostic_summary_metrics.json`, `runs_wave2/`):** Fixed thin-barrier clearance (`41.7%–48.6%`), positive persistence skill (`+10.8%` to `+38.0%`), and `152M`/`453M` MPC (`66.7%` / `62.5%`), **and uncovered two clean deep-scaling mathematical pathologies at `L = 23` (`SLWM-1B`):**
   - *Pathology A (Cascaded 18-Layer IIR Sensory Phase Lag):* Passing `SLOTS_VISUAL` (`0..15`) and `SLOT_PROPRIO` (`18`) purely through `c_next = gamma * c_prev + (1 - gamma) * u_t` with zero instantaneous `u_t` skip introduced 2–4 ticks of group delay across `L = 23` layers, causing `SLWM-1B` to overshoot targets (`35.4%` MPC vs `70.8%` on zero-lag `DenseTransformer-1B`). Confirming this, `abl_no_registers` (which bypassed `c_prev` on `SLOTS_REGISTER`) jumped to `73.6%` MPC in Wave 2.
   - *Pathology B (Visual Pose Drift on Symbolic Keys & Register Variance in `txt_ctx`):* At `T = 512`, Wave 2 `SLWM-1B` scored `80.0%` (`8/10`) on un-overwritten `k1_init` but `0.0%` (`0/6`) on overwritten `k0_over`, whereas `abl_separate_cores` scored **`100.0%` (`16/16`) across all 3 seeds**! Root cause: full bidirectional attention from 16 moving visual patches into `SLOT_TEXT` shifted $k_w(t_{\text{ov}}) \neq k_w(1)$ so Householder erasure $(I - \beta_t k_t k_t^\top)$ missed the initial key, while `txt_ctx = 0.50 * z[SLOT_TEXT] + 0.50 * reg_sum` injected physical register variance into `text_logits`.
3. **Wave 3 (`summary_metrics.json`, `runs/`, `figures/`):** Implemented zero-parameter **Zero-Lag Multi-Rate `GatedDelta`** + directed cross-modal grounding (`_shield_text_slot_mask`: embodied/register slots attend freely to `SLOT_TEXT`, while `SLOT_TEXT` remains symbol-pure; verified locally to have `0.00e+00` visual pose drift and `100.9%` tick-instantaneous step response at `L = 23`). Non-`GatedDelta` baselines (`dense_1b`, `legacy_1b`, `abl_dense_attn`, `abl_scalar_retention`) had `0`-bit code diff and were verified-reused directly; all `20` `GatedDelta` runs (`1.008B`, `453M`, and `18 × 152M`) were re-trained and evaluated end-to-end.

---

## 3. Strict Parameter Accounting, Config Hashes & Dual-HDD Receipts

| Model / Cohort Group | Arch | Strict Total Params | Adapters | Processor Core | JEPA & Unc | Policy & Heads | Config SHA-256 (`s2001`) | `int8` Portable SHA-256 (`s2001`) | Accelerator |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- | :--- |
| **`SLWM-X1B` (`flagship_1b`)** | `slwm_x1b` | **`1,008,426,726`** | `19,033,728` | `954,618,720` | `16,529,666` | `18,244,612` | `27bc1731ecbf` | `70f7565b4672` | TPU `v6e-1` |
| **`DenseTransformer-X1B` (`dense_1b`)** | `dense_transformer` | **`1,008,407,430`** (`-0.0019%`) | `19,033,728` | `954,599,424` | `16,529,666` | `18,244,612` | `8a04d4b49755` | `61db03d99eba` | TPU `v6e-1` |
| **`LegacyScalarHybrid-X1B` (`legacy_1b`)** | `legacy_scalar_hybrid` | **`1,008,407,430`** (`-0.0019%`) | `19,033,728` | `954,599,424` | `16,529,666` | `18,244,612` | `e73c52c71b01` | `f07bb3432a69` | TPU `v6e-1` |
| **`SLWM-X450M` (`scale_450m`)** | `slwm_x1b` | **`453,179,166`** | `11,769,408` | `420,419,136` | `9,375,170` | `11,615,452` | `5be1bb5af10a` | `ed97d2b6ae33` | TPU `v6e-1` |
| **`SLWM-X150M` (`abl_slwm_full`, 3 seeds)** | `slwm_x1b` | **`152,003,766`** | `6,401,856` | `134,702,832` | `4,430,978` | `6,468,100` | `5b5c651c12c9` | `d029e93be255` | `A100-40GB` |
| **`152M` GatedDelta Ablations (5 × 3 seeds)** | `slwm_*` / `separate` | **`152,003,766`** (`0.0000%`) | `6,401,856` | `134,702,832` | `4,430,978` | `6,468,100` | per-variant | dual-HDD verified | `A100-40GB` |
| **`152M` Attn / Scalar Ablations (2 × 3 seeds)** | `dense` / `scalar` | **`151,995,846`** (`-0.0052%`) | `6,401,856` | `134,694,912` | `4,430,978` | `6,468,100` | per-variant | dual-HDD verified | `A100-40GB` |

### Datasets, Sample Counts & Zero-Leakage Modality Mixture (`x1b_real_bank_v1.card.json`)
- **Real Multimodal Bank (`x1b_real_bank_v1.npz`, SHA-256 `64817dbd9e55...`):** Strictly disjoint train/held-out-eval splits staged from real GreatestHits impact audio-video, STARSS23 spatial audio-video, DROID robot manipulation trajectories, and FineWeb subword language windows (`vocab_size = 4096` role-structured subtokens, zero hash collisions), plus `x1b_pymunk_panels_64.npz` (`SHA-256 531c84f040d7...`, `64` held-out `64×64` PyMunk `puck_barrier` + `pusht` episodes).
- **Online Causal Physics & Associative Curriculum (`data/x1b_multimodal.py`):** 50%/50% mixture per `batch_size = 12, seq_len = 32` step (`2,600` steps = `31,200` sequences at `1.008B`/`453M`; `1,600` steps = `19,200` sequences at `152M`) covering open PushT, `1-px` central thin-barrier navigation, `0.4s` transient goal commands, 180° inverted-camera spoofing (`t >= 4`), ungrounded sensor dropout, and multi-query `16`-key / `16`-value associative overwrite streams.

---

## 4. Primary Quantitative Results (All 28 Verified Runs)

### 4.1 1.008B Head-to-Head & Scaling Trajectory (`152M → 453M → 1.008B`)

| Model (`Strict Params`) | Closed-Loop MPC | Open PushT MPC | `1-px` Barrier MPC | G8 3-Shot | Assoc `T=32` | Assoc `T=256` | Assoc `T=512` (16×) | Real AV `R@1` | 1-Step / 4-Step Skill | Latent `PR` | FineWeb PPL | Spoof Regret (Learned / Always) | Unsupp. Claim |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`SLWM-X1B` (`1.008B`)** | **`77.1%`** | **`100.0%`** | **`54.2%`** | `41.7%` | **`100.0%`** | **`100.0%`** | **`100.0%`** | `90.6%` | `+10.0%` / `+16.7%` | **`0.107`** | `864.9` | **`0.306`** / `0.404` | **`0.0%`** |
| **`DenseTransformer-X1B` (`1.008B`)** | `70.8%` | `100.0%` | `41.7%` | `18.8%` | `100.0%` | `6.2%` | `37.5%` | `92.2%` | `+11.5%` / `+17.5%` | `0.037` | `1407.2` | `0.370` / `0.408` | `0.0%` |
| **`LegacyScalarHybrid-X1B` (`1.008B`)** | `68.8%` | `100.0%` | `37.5%` | `31.2%` | `6.2%` | `0.0%` | `6.2%` | `1.6%` | **`+38.0%` / `+32.8%`** | `0.040` | **`845.3`** | `0.288` / `0.453` | `0.0%` |
| **`SLWM-X450M` (`453M`)** | **`79.2%`** | **`100.0%`** | **`58.3%`** | `27.1%` | **`100.0%`** | **`100.0%`** | **`100.0%`** | **`93.8%`** | `+12.2%` / `+22.1%` | **`0.120`** | `865.6` | `0.299` / `0.404` | **`0.0%`** |
| **`SLWM-X150M` (`152M`, 3 seeds)** | `73.6 ± 1.0%` | `97.2 ± 2.0%` | `50.0 ± 0.0%` | `34.0 ± 1.0%` | `100.0 ± 0.0%` | `100.0 ± 0.0%` | `100.0 ± 0.0%` | `90.6 ± 3.4%` | `+12.5%` / `+19.5%` | `0.103` | `430.6` | `0.342` / `0.460` | `0.0%` |

### 4.2 152M 3-Seed Factorial Ablation Matrix (`24` Runs, Mean ± SD)

| Factorial Variant (`152M`, `n=3` seeds) | Isolates Bottleneck | Closed-Loop MPC | `1-px` Barrier MPC | Assoc `T=256` | Assoc `T=512` (16×) | Real AV `R@1` | 1-Step Skill | Latent `PR` | Transient Cue Drop (`pp`) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`abl_slwm_full` (Full SLWM)** | Reference unified design | **`73.6 ± 1.0%`** | **`50.0 ± 0.0%`** | **`100.0 ± 0.0%`** | **`100.0 ± 0.0%`** | `90.6 ± 3.4%` | `+12.5 ± 2.0%` | **`0.103`** | `+4.2 ± 10.3` |
| **`abl_no_registers`** | Removes `SLOTS_REGISTER` & write routing | `72.9 ± 3.0%` | `45.8 ± 5.9%` | `6.2 ± 0.0%` | **`10.4 ± 3.0%`** | `92.2 ± 1.3%` | `+18.2 ± 1.5%` | `0.101` | `-10.4 ± 15.0` |
| **`abl_scalar_retention`** | Replaces matrix `GatedDelta` w/ scalar EMA | `67.4 ± 6.4%` | `44.4 ± 7.9%` | `6.2 ± 5.1%` | **`4.2 ± 3.0%`** | `92.7 ± 1.5%` | `+18.2 ± 1.2%` | `0.024` | `-11.8 ± 7.4` |
| **`abl_dense_attn`** | Replaces `GatedDelta` w/ full RoPE attention | `70.8 ± 5.1%` | `41.7 ± 10.2%` | **`35.4 ± 7.8%`** | `62.5 ± 0.0%` | `88.5 ± 2.7%` | `+10.8 ± 4.0%` | `0.025` | `-11.8 ± 2.0` |
| **`abl_separate_cores`** | Blocks cross-modal spatial attention | **`56.9 ± 7.1%`** | **`25.0 ± 10.2%`** | `100.0 ± 0.0%` | `100.0 ± 0.0%` | **`63.0 ± 5.9%`** | `+25.9 ± 2.0%` | `0.025` | `+6.2 ± 1.7` |
| **`abl_raw_pixel_mse`** | Replaces EMA-JEPA w/ pixel/audio linear MSE | `71.5 ± 5.9%` | `45.8 ± 8.5%` | `100.0 ± 0.0%` | `100.0 ± 0.0%` | **`1.0 ± 0.7%`** | `+16.8 ± 2.5%` | **`0.009`** | `+2.1 ± 8.5` |
| **`abl_direct_jepa_no_res`** | Removes Euler residual latent integration | **`32.6 ± 8.8%`** | **`20.8 ± 15.6%`** | `100.0 ± 0.0%` | `100.0 ± 0.0%` | `92.2 ± 2.2%` | **`-9.9 ± 3.5%`** | `0.018` | `+25.0 ± 15.6` |
| **`abl_jepa_no_sigreg`** | Removes `SIGReg` variance-covariance loss | `75.7 ± 8.5%` | `58.3 ± 15.6%` | `100.0 ± 0.0%` | `100.0 ± 0.0%` | `96.9 ± 1.3%` | `+16.7 ± 3.3%` | **`0.011`** | `+19.4 ± 10.0` |

### 4.3 Paired Bootstrap 95% Confidence Intervals (`10,000` Resamples)

| Paired Contrast | Mean Effect ($\Delta$) | 95% Bootstrap CI | Interpretation |
| :--- | ---: | :---: | :--- |
| **`1B` `SLWM-X1B` vs `LegacyScalarHybrid-1B` (`MPC Success`)** | **`+8.33 pp`** | `[+2.08 pp, +16.67 pp]` | Statistically significant `1B` closed-loop control gain (`CI > 0`) |
| **`1B` `SLWM-X1B` vs `DenseTransformer-1B` (`MPC Success`)** | **`+6.25 pp`** | `[-2.08 pp, +16.67 pp]` | Positive `+6.25 pp` overall (`+12.5 pp` on `1-px` thin-barrier tasks) |
| **`1B` `SLWM-X1B` Latent-CEM-MPC vs G8 3-Shot Shooting** | **`+35.42 pp`** | `[+22.92 pp, +47.97 pp]` | Decisive planning gain on identical `1.008B` weights (`CI >> 0`) |
| **`1B` `SLWM-X1B` Latent-CEM-MPC vs Open-Loop Policy** | **`+47.92 pp`** | `[+33.33 pp, +62.50 pp]` | Receding-horizon planning resolves open-loop compounding error |
| **`1B` `SLWM-X1B` vs `DenseTransformer-1B` (`T=512` Overwrite)** | **`+62.50 pp`** | `[+37.50 pp, +87.50 pp]` | Eliminates RoPE OOD horizon failure (`100.0%` vs `37.5%`; `+93.75 pp` at `T=256`) |
| **`1B` `SLWM-X1B` vs `LegacyScalarHybrid-1B` (`T=512` Overwrite)** | **`+93.75 pp`** | `[+81.25 pp, +100.00 pp]` | Resolves scalar-retention overwrite impossibility (`100.0%` vs `6.25%`) |
| **`1B` `SLWM-X1B` vs `LegacyScalarHybrid-1B` (`Real AV R@1`)** | **`+89.06 pp`** | `[+79.69 pp, +96.88 pp]` | Resolves 1-vector cross-modal collapse (`90.63%` vs `1.56%`) |
| **`152M` Pooled `SLWM-Full` vs `Direct-JEPA-No-Res` (`MPC`)** | **`+40.97 pp`** | `[+33.33 pp, +48.63 pp]` | Proves Euler residual latent integration is essential (`73.6%` vs `32.6%`) |
| **`152M` Pooled `SLWM-Full` vs `Separate-Cores` (`MPC` / `AV R@1`)** | **`+16.67 pp` / `+27.60 pp`** | `[+9.72, +23.61]` / `[+20.83, +34.90]` | Proves shared cross-modal spatial field drives both control and alignment |
| **`152M` Pooled `SLWM-Full` vs `Scalar-Retention` (`T=512`)** | **`+95.83 pp`** | `[+89.58 pp, +100.00 pp]` | 3-seed confirmation of Householder `GatedDelta` overwrite recall |
| **`152M` Pooled `SLWM-Full` vs `Raw-Pixel-MSE` (`Real AV R@1`)** | **`+89.58 pp`** | `[+84.90 pp, +93.75 pp]` | Proves latent-JEPA + alignment outperforms raw-pixel MSE on grounding |

---

## 5. Spatial Barrier Resolution, CommitGate Safety & Latency Scaling

- **Pre-Impact `1-px` Thin-Barrier Detection (`Panel 2`):** All `64×64` `SLWM` models achieve **`1.000` AUROC** both on synthetic `1-px` barriers (ticks `0..2` prior to contact, with contact bits zeroed) and on held-out `64×64` PyMunk `puck_barrier` vs `pusht` scenes (`pymunk_barrier_auroc = 1.000`), whereas degrading the input to G8-style `16×16` stride-4 subsampling drops pre-impact AUROC to **`0.583`** (`SLWM-X1B`) and `0.235` (`DenseTransformer-1B`), directly confirming that `16×16` spatial subsampling was the root cause of G8 barrier blindness.
- **Adversarial Camera Spoofing & Hallucination Gate (`Panel 6`):** On `SLWM-X1B`, the learned innovation `PolicyCommitGate` achieves `77.4%` spoof-mode switching (`95.3%` at `453M`), **`96.4%`** hidden-uncertainty probing/abstention when sensory grounding is masked (`0.0%` unsupported claims vs `100.0%` under `always_commit`), well-calibrated uncertainty (`ECE = 0.0321` vs `0.0590` on `DenseTransformer-1B`), and `67.1%` clean-task usefulness (zero abstention collapse).
- **Throughput & Memory Complexity (`Panel 7`):** On TPU `v6e-1`, `SLWM-X1B` processes `T = 32` (`768` spatiotemporal slots) in **`9.75 ms`** and `T = 256` (`6,144` slots) in **`113.91 ms`** (`107,878` slots/s) with strictly bounded $\mathcal{O}(1)$ recurrent state memory (`H \times D_h^2 + S \times D = 135,168` floats/layer = `12.4 MB` total across all `23` layers, independent of sequence horizon $T$).

---

## 6. Failure Modes, Honest Trade-Offs & Boundaries (`AGENTS.md` §13)

1. **Reactive Amortized Policy Latching Under Fixed Step Budgets (`Panel 4`, `1.008B` vs `453M`):** While `SLWM-X450M` (`453M`) achieves **`93.8%` reactive goal completion** after a `0.4s` transient cue vanishes (`-8.3 pp` drop vs `85.4%` continuous — completely eliminating `LegacyScalarHybrid-1B`'s `+70.8 pp` amnesia!), `SLWM-X1B` (`1.008B`) trained for the identical `2,600` steps (`31,200` episodes, `2.23×` fewer tokens per parameter) exhibits under-converged reactive `action_proposal` latching (`39.6%` continuous → `2.1%` transient, `+37.5 pp` drop) when evaluated *without* Latent-CEM-MPC. With receding-horizon Latent-CEM-MPC (`Panel 5`), `SLWM-X1B` achieves `77.1%` closed-loop success (`100.0%` open PushT). Scaling `1.008B` reactive policy distillation proportionally to Chinchilla token-per-parameter parity (`~6,000` steps) is the primary lever to bring `1.008B` reactive `action_proposal` up to `SLWM-X450M`'s `93.8%`.
2. **1-Step Unimodal Proprioceptive MSE vs Multimodal Latent Objective (`Panel 2`):** `LegacyScalarHybrid-1B` (`raw_pixel_linear_mse`, which drops `jepa_loss`, `sigreg`, and `av_nce` saturation) attains higher 1-step synthetic coordinate skill (`+38.0%` vs `+10.0%` 1-step / `+16.7%` 4-step on `SLWM-X1B`), at the cost of complete cross-modal collapse (`1.6%` AV `R@1`), `0%–6.2%` associative recall, `+70.8 pp` transient-command amnesia, and `-8.3 pp` worse closed-loop MPC (`68.8%` vs `77.1%`).
3. **Scope Boundary:** Although `x1b_real_bank_v1` evaluates real DROID, GreatestHits, STARSS23, and FineWeb windows, closed-loop manipulation rollouts (`Panels 4–6`) execute in deterministic `64×64` rigid-body contact physics rather than on physical robot hardware. No claims of human-level physical understanding or unconstrained real-robot safety are made beyond these controlled benchmarks.

---

## 7. Next Allowed Step

1. Scale `SLWM-X1B` training horizon from `2,600` to `6,000–10,000` token-matched steps on multi-chip TPU `v6e` with richer real DROID + OXE manipulation action supervision to close the `453M → 1.008B` reactive policy-head Chinchilla gap while preserving `100%` `T = 512` associative recall and `77.1%+` Latent-CEM-MPC.
2. Validate `int8`-quantized `SLWM-X450M` (`415 MB`) and `SLWM-X1B` (`919 MB`) streaming inference (`init_streaming_memories`) in a real-time hardware-in-the-loop camera + microphone + arm controller testbed with `PolicyCommitGate` active.
