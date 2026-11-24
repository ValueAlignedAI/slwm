# Signal-Latent World Models (`SLWM-v3`): Selective Multi-Timescale Matrix Memory, Causal Streaming Codecs, and Multimodal Scaling from 42M to 1.07B Parameters

**Iteration / Workstream:** `SLWM-v3` (`research/slwm-v3-selective-memory`, Waves A–D + `v2` Epoch-Capped Multi-Domain Scaling + Frozen-Weight Long-Horizon Audit)  
**Repository Provenance:** Branch `research/slwm-v3-selective-memory`, evaluated at commit `fbbe469` (Gate 3 fingerprint at commit `4f9387c` in `reports/v3/gate3_architecture_and_mup_receipt.json` covers all 6 core backbone/spec/memory/objective files unchanged through `fbbe469`; two subsequent commits modified only `models/v3/any2any_heads.py` to add `scratch_edge` audio readouts and Wave C SwiGLU/spatial-refiner decoders)  
**Parameter Accounting Modes Reported:** Dual-Track (`Core Comparison` and `Strict System Cost` per `AGENTS.md §4.1`)  
**Evaluation Scope:** 43 core training runs total — **28** evaluated on sealed test partitions (`12` `2×2` representation ablations, `7` `137.8M v2`, `6` `703.1M v2`, `2` `1.072B v2`, `1` `1.072B v1` baseline) and **15** `42.4M` Real-Speech Overwrite (`L12_Weq64_additive`) runs evaluated on speaker-disjoint dev splits — plus `15` frozen-core `PolicyCommitGate` (Wave B, dev) evaluations, `15` frozen-core multimodal decoder (Wave C, dev) audits, and `5` frozen-weight `> 1.0 s` local long-horizon evaluations on sealed test episodes (`reports/v3/v2_long_horizon_frozen_eval_receipt.json`).

---

## 1. Executive Summary & Falsifiable Hypotheses

We investigate whether a strictly causal, bounded-memory ($\mathcal{O}(1)$ streaming state) **Signal-Latent World Model (`SLWM-v3`)**—combining short-window Grouped-Query Attention (GQA) with surprise-routed multi-timescale gated matrix memory (`ContentRoutedMultiTimescaleMemory`)—improves multimodal signal prediction, cross-modal causal grounding, selective long-horizon memory retention, and epistemic commitment control relative to parameter- and byte-matched recurrent and sliding-window attention baselines across four scales (`42.4M`, `137.8M`, `703.1M`, and `1.072B` trainable core parameters).

### Falsifiable Research Hypotheses

- **H1 (Selective Multi-Timescale Memory & Routing):** Under matched parameter budgets and bounded $\mathcal{O}(1)$ streaming state, pairing local sliding-window GQA with multi-timescale gated matrix memory (`SLWM-v3`) outperforms equal-byte `Sliding-Window GQA` on long-horizon overwrite recall (`167.0 s` / `2,088` ticks on `L12_Weq64_additive`, strictly beyond the `discovery_45m` GQA stacked receptive field `RF = 756`), language/code modeling, and physical audio-visual prediction, while surprise-based content-to-timescale routing (`slwm_v3_routed`) improves epistemic commitment and long-horizon overwrite over uniform bank-averaged routing (`slwm_v3_no_routing`) and single-bank `Gated DeltaNet`.
- **H2 (Epoch-Capped Multi-Domain Scaling without Memorization):** Replacing small-shard multi-epoch looping (`v1`, up to `88×` passes) with expanded, train-to-eval media-disjoint (`0` train-to-dev and `0` train-to-test video/audio/trajectory overlap, `0.0%` exact and 64-byte text contamination) 8-domain corpora under temperature-smoothed epoch-capped sampling (`max_epochs <= 5.0`, `v2`) eliminates cross-modal overfitting and yields monotonic sealed-test gains from `137.8M` to `703.1M` and `1.072B` parameters.
- **H3 (Representation Attribution — `Backbone × Representation`):** A `2×2` factorial separating sequence backbone (`slwm_v3_routed` vs. `single_bank_gated_deltanet`) from input edge representation (`scratch_edge` causal 64-bin log-mel vs. `frozen_codec_edge` Kyutai Mimi 32-RVQ) disambiguates backbone effects from pretrained codec representations.
- **H4 (Frozen-Core Commitment & Cross-Modal Generation):** Operating strictly on `stop_gradient(Z_t)` without updating world-core weights, a decoupled `PolicyCommitGate` reduces unsupported claims under zero-shortcut blind reliability (`reliability == 1.0` identically across supported and unsupported trials) without collapsing to abstention, and decoupled multimodal decoders exhibit verified causal conditioning under condition-shuffling and action-intervention tests.

### Summary of Empirical Outcomes

1. **Decisive Advantage Over Equal-Byte `Sliding-Window GQA` Across Scales:** At both `137.8M` and `703.1M` (`n=2` seeds), `SLWM-v3` recurrent cores outperform parameter- and byte-matched `Sliding-Window GQA` on WikiText-103 sealed test (`1.945` vs. `1.982` BPB at `703.1M`; `1.971` vs. `2.032` BPB at `137.8M`), LAMBADA final-word BPB (`1.657` vs. `2.055` at `703.1M`; `1.731` vs. `2.466` at `137.8M`), physical-impact `Video -> Audio` causal gain on GreatestHits (`+0.0301` vs. `+0.0167` nats, `1.8×` stronger at `703.1M`), ESC-50 `Audio -> Text` causal gain (`+0.0935` vs. `+0.0492` nats, `1.9×` stronger at `703.1M`), Wave C robotics next-video Huber (`0.1263` vs. `0.1400` at `703.1M`), and `167.0 s` Real-Speech Overwrite `L12_Weq64_additive` (`13.54%` vs. `2.08%`, `+11.5 pp`, 95% CI `[+4.1, +19.0]`, dev split). **Honest GQA wins:** at `703.1M`, `Sliding-Window GQA` edges `SLWM-v3` on LAMBADA exact-word match (`6.38%` vs. `5.38%`), ESC-50 Fold-5 linear probe (`61.88%` vs. `59.38%`), MSR-VTT caption F1 (`0.1931` vs. `0.1913`), and open-loop blind-action `video2action` motor L1 (`0.294` vs. `0.364`).
2. **Nuanced Standing Against `Single-Bank Gated DeltaNet` and `No-Routing`:**
   - Against faithful `single_bank_gated_deltanet`, `slwm_v3_routed` wins broadly at `137.8M` (`n=2`: WikiText `1.971` vs. `2.001` BPB, LAMBADA `4.65%` vs. `3.49%`, Wave B blind pass `2/2` vs. `0/2`, Wave C robotics video Huber `0.1040` vs. `0.1145`) and re-establishes a clear lead at `1.072B` (`n=1`: WikiText `1.935` vs. `1.949` BPB, LAMBADA `6.73%` vs. `5.55%`, MBPP `1.226` vs. `1.263` BPB, GreatestHits `+0.0354` vs. `+0.0284`, MSR-VTT F1 `0.1797 (+0.0183)` vs. `0.1447 (-0.0008)`). At `703.1M` (`n=2`), the two recurrent architectures split: `slwm_v3_routed` leads MBPP/HumanEval BPB, ESC-50 MLP/A2T, VGGSound CE, Wave B (`+4.91` vs. `+4.69 pp`), and Wave C robotics video Huber (`0.1263` vs. `0.1402`), whereas `single_bank` edges WikiText (`1.940` vs. `1.945`), LibriSpeech A2T (`+0.0759` vs. `+0.0553`), and `42.4M` RSO `L12_Weq64_additive` dev (`16.15%` vs. `13.54%`, `-2.6 pp` 95% CI `[-12.5, +6.5]`, statistically indistinguishable) with `1.67×` smaller core streaming state (`4.50` vs. `7.50 MiB` at `703.1M`) and `1.77×–1.84×` faster training wall time (`1.81×` mean across seeds and mixed accelerators).
   - In the single-factor routing ablation at `137.8M` (`n=2`, exact equal `3.00 MiB` state and FFN width), `slwm_v3_routed_c16` matches `slwm_v3_no_routing_c16` on WikiText (`1.9706 ± 0.0036` vs. `1.9710 ± 0.0100`) and GreatestHits (`+0.0298 ± 0.0004` vs. `+0.0296 ± 0.0053`), uniquely passes Wave B blind epistemic commitment `2/2` (`+5.35 ± 0.22 pp` vs. `+4.40 ± 0.98 pp`), and improves `42.4M` RSO `L12_Weq64_additive` overwrite (`13.54%` vs. `9.90%`, Overwrite A/B `12.28%` vs. `7.02%`), while `no_routing` edges LAMBADA (`4.99 ± 0.02%` vs. `4.65 ± 0.15%`), HumanEval BPB (`1.081` vs. `1.110`), and LibriSpeech greedy WER (`1.38` vs. `1.56`)—with `n=2` spread favoring `routed` on WikiText/GreatestHits and `no_routing` on LAMBADA/MBPP.
3. **Data & Schedule Scaling (`v1 -> v2`) Outweighs Parameter Count Alone:** Moving from the mid-cosine `1.072B v1` checkpoint (`98.3M` positions, `6,000/18,000` steps, 7 domains with heavy AV looping) to `137.8M v2` (`294.9M` positions, full cosine, 8 domains, `max_ep <= 5.0`) improves WikiText-103 BPB from `2.317` to `1.971`, LAMBADA accuracy from `0.27%` to `4.65%`, ESC-50 Fold-5 linear accuracy from `51.75%` to `59.88%`, and VGGSound V2A cross-entropy from `9.344` (overfit collapse) to `6.008` using `7.8×` fewer parameters (while noting in Table 3.2 that `1.072B v1` still scored higher on Clotho F1, ESC-50 A2T, and robotics CF gain due to heavy multi-epoch passes on those smaller `v1` shards).
4. **Representation Attribution (`2×2` Factorial, `n=3`):** On ESC-50 Fold-5 linear classification, zero-pretrained causal 64-bin log-mel (`scratch_edge`) reaches `62.42 ± 1.66%` (`routed`) / `61.42 ± 1.01%` (`single_bank`), beating frozen Mimi 32-RVQ (`43.83 ± 2.50%` / `43.67 ± 0.29%`) by `+18.6 pp`, while frozen Mimi dominates Clotho caption F1 (`0.2016` vs. `0.0594`). Thus acoustic classification gains do not require a pretrained audio codec, whereas semantic audio-to-text captioning at low data budgets benefits strongly from Mimi's speech/audio pretraining.
5. **Major Unresolved Limitations & Negative Results (Reported Prominently in Primary Tables):**
   - `HumanEval` and `MBPP` executed `pass@1 = 0.0%` across all models and scales (`0.17–0.26B` text/code bytes is far below functional Python synthesis).
   - Zero-shot `Real-Speech Overwrite L16_Weq128_additive` (`274.0 s`) counterfactual flip accuracy is `0.0%` without task supervision.
   - Greedy `T=0` LibriSpeech byte ASR collapses into unconditioned repetition loops (`WER = 1.38–1.69 > 1.0`) despite positive teacher-forced `Audio -> Text` causal shuffle gains.
   - Wave B `PolicyCommitGate` reduces unsupported claims by `55.3–64.7 pp` arm-mean (`54.2–68.2 pp` per seed) blind and `87.5–95.6 pp` arm-mean (`86.4–96.5 pp` per seed) in degraded outage, but exhibits high Expected Calibration Error (`ECE = 0.385–0.416` blind; `0.224–0.279` degraded) and passes the strict `> +5.0 pp` blind gain gate over `reliability_only` in only `4/15` runs (`2/2` on `137.8M routed`, `1/2` on `137.8M no_routing`, `1/2` on `703.1M routed`, `0/9` on `single_bank`, `sw_gqa`, and `1.072B`).
   - At `1.072B`, the hard `5.0`-epoch cap restricts robotics exposure to `1.07M` ticks (`0.001` ticks/param), causing robotics Wave C video Huber degradation (`0.104 -> 0.126 -> 0.160`) and free-running `1.80 s` `action2video` rollout collapse (`10.44 dB` vs. `18.60–20.47 dB` at `137.8M`/`703.1M`).
   - In `> 1.0 s` local generation, deterministic Huber video heads produce mean-of-modes fog on under-determined `audio2video`/`text2video` (`12.1–12.7 dB`), open-loop `video2audio` lacks frame-interleaved synchronization (`env_r = -0.11..+0.08`) and exhibits `46.9%–91.8%` RVQ-0 adjacent repetition (`89.8%–91.8%` on `slwm_v3_routed`; `93.9%–100.0%` on `audio2audio`), and the `video2text` evaluation script truncates visual prefixes at `min(ctx_end, 512)`.

### Scorecard Against `AGENTS.md §15` Project Success Criteria

| `AGENTS.md §15` Criterion | Verdict | Evidence & Caveats |
|---|---|---|
| 1. Advantage over GPT-2-style / attention baselines on periodic/audio/visual signal prediction | **PASS (vs. eq-byte SW-GQA)** | `1.8×` stronger GreatestHits physical-impact V2A (`+0.0301` vs. `+0.0167`), `1.9×` stronger ESC-50 A2T (`+0.0935` vs. `+0.0492`), lower Wave C robotics video Huber (`0.1263` vs. `0.1400`), and `+11.5 pp` on `167.0 s` RSO `L12_Weq64_additive` dev (`13.54%` vs. `2.08%`). *Caveat: no dense quadratic attention baseline at `137.8M–1.072B` due to non-$\mathcal{O}(1)$ KV scaling; GQA leads ESC-50 linear probe at `703.1M`.* |
| 2. Competitive text/code performance | **PARTIAL** | Beats equal-byte SW-GQA on WikiText-103 BPB (`1.945` vs. `1.982`), LAMBADA BPB (`1.657` vs. `2.055`), and HumanEval/MBPP BPB at `137.8M` and `703.1M`; however, `0.17–0.26B` byte-level text/code exposure leaves `HumanEval/MBPP pass@1 = 0.0%` and WikiText PPL (`1,260–1,428`) far behind `10B–300B`-token BPE LMs. |
| 3. Measurable cross-modal transfer / causal grounding | **PASS (with domain exceptions)** | Statistically verified causal conditioning (`15/15` sealed-test runs) on GreatestHits V2A, LibriSpeech A2T, ESC-50 A2T, and robotics action-to-proprio/video. *Partial / negative domains:* Clotho A2T verifies on `5/15` sealed-test runs (`10/15` with `CI95_low > 0`, `15/15` positive point gain, `13/15` verified on dev); VGGSound V2A shuffle gain is near zero (`-0.002` to `+0.008` nats); MSR-VTT caption margin is inconsistent (`-0.005` to `+0.018`). |
| 4. Lower unsupported-output rate at similar usefulness | **PARTIAL** | Wave B cuts unsupported claims by `55.3–64.7 pp` arm-mean (`54.2–68.2 pp` per seed) blind and `87.5–95.6 pp` arm-mean (`86.4–96.5 pp` per seed) degraded while preserving `69.16–75.80%` arm-mean committed content accuracy (`68.13–75.80%` per seed vs. `66.70–69.34%` per-arm always-commit baselines) without abstention collapse (`49.8–64.5%` abstention); however, blind `ECE` remains poor (`0.385–0.416`) and blind accuracy beats `reliability_only` by only `+2.6` to `+5.6 pp`. |
| 5. Interpretable policy behavior (`speak/act/wait/no-op`) | **PARTIAL** | Clear separation between `SPEAK_TEXT` and `ABSTAIN_SILENCE` under blind conflict and sensor outage (`action_distribution` tracked in receipts); `EMIT_AUDIO` and `ACT_MOTOR` are `0.0%` (`GENERATE_VIDEO <= 0.49%`) because the current Wave B harness excludes robotics and masks evaluation to post-text queries. |
| 6. Diagnostic evidence of cross-modal latent structure | **PASS** | Linear/MLP probes recover 50-class ESC-50 (`57–64%` vs. `2%` chance), `text2audio` prompt conditioning shifts spectral centroid by `2,508–4,020 Hz` (with stationary RVQ-0 caveat in `§6.3`), and `703.1M` `action2video` rollouts show `+4.10` to `+5.60 dB` counterfactual separation (`1.80 s`). |
| 7. Ablation evidence that shared latent signal design matters | **PARTIAL** | Clear advantage of recurrent latent memory over equal-byte SW-GQA (`+11.5 pp` RSO `L12_Weq64_additive`, consistent `137.8M`/`703.1M` gains) and of dual-bank routing over `no_routing` on Wave B pass rate (`2/2` vs. `1/2`) and RSO (`+3.6 pp`); however, `single_bank_gated_deltanet` remains highly competitive at `703.1M` and `42.4M` RSO, and `no_nowrite`/`no_zerolag`/`tied_tau` were not re-run at `v2` confirmation scale. |

---

## 2. Architecture, Mathematical Formulation & Dual-Track Cost Accounting

### 2.1 Strictly Causal Multimodal Packet Stream & Core Architecture

All compared backbones share an identical strictly causal packet interface operating on a unified latent world field $Z_t \in \mathbb{R}^{d}$ (`models/v3/spec.py`, `models/v3/core.py`, `models/v3/memory.py`). Five modality channels (`0: video`, `1: audio`, `2: text/code UTF-8 byte`, `3: motor action + proprioception`, `4: signal probe`) are projected into $\mathbb{R}^d$ alongside elapsed physical time $\Delta t_t \ge 0$, validity mask $m^{\text{vld}}_t \in \{0, 1\}$, and sensor reliability $r^{\text{rel}}_t \in [0, 1]$.

Each of the $L$ processor layers (`SLWM-v3`) applies three pre-RMSNorm residual sublayers with CompleteP ($\alpha_{\text{CP}} = 1.0$) depth residual scaling $\gamma_L = 0.25 \cdot (L_{\text{base}} / L)$ (`L_base = 12`) and base-shape $\mu\text{P}$ width scaling ($m_d = d / d_{\text{base}}$, $d_{\text{base}} = 256$). Exact $\mu\text{P}$ parameter roles (`models/v3/memory.py:116-143`, `models/v3/core.py:482-496`) assign `lr_mult = 1 / m_d` to width-scaling `hidden` and `readout` matrices, `lr_mult = 128 / fan_in` to `fixed_hidden` memory output projections (`w_o`, `w_prior_proj`, yielding `0.5` at `137.8M` and `0.25` at `703.1M`/`1.072B`), and `lr_mult = 1.0` to `input` and `scalar` parameters—with `weight_decay = 0.0` on all scalar-role leaves (1D norms/biases/timescales plus 2D scalar-role `o_norm_w`, `rel_pos_bias`, `g_bias`). After the final layer norm, an optional zero-lag residual bypass computes $z_t = \mathrm{RMSNorm}(h_t^{(L)}) + \sigma(g_{z_0})(x_{\text{raw},t} W_{z_0})$ (`models/v3/core.py:360-361`). Coordinate stability was verified across widths $d \in \{128, 256, 512\}$ and depths $L \in \{4, 6, 8\}$ (`reports/v3/gate3_architecture_and_mup_receipt.json`).

1. **Short-Window $\mathcal{O}(1)$ Causal Local GQA Ring Buffer:** Fixed window $W$ in `bfloat16` (`K, V` with $H_{\text{kv}}$ KV heads and $H_q$ query heads, head dim $d_h = 64$, learned relative position bias). Maximum stacked local receptive field is $\mathrm{RF}_{\text{local}} = L(W - 1)$.
2. **Content-Routed Multi-Timescale Selective Matrix Memory (`slwm_v3_routed`):** Maintains $B = 4$ timescale banks with $H_m$ heads per bank ($H_{\text{tot}} = B H_m$) and `float32` associative state matrices $M_{t}^{(b,h)} \in \mathbb{R}^{d_v \times d_k}$. Let $\tilde{x}_t = \mathrm{RMSNorm}(x_t)$ and $\Delta t_t^+ = \max(\Delta t_t, 0)$.
   - **Octave-banded physical decay:**
     $$\tau_b = \exp\!\left(\ln \tau_b^{(0)} + 0.7 \tanh\!\left(\theta_{\tau,b} - \ln \tau_b^{(0)}\right)\right), \quad \tau^{(0)} = (1, 8, 64, 512)\text{ s}$$
     $$g_t^{(b,h)} = \sigma\!\left(\tilde{x}_t W_g + b_g\right)_{b,h}, \qquad \alpha_t^{(b,h)} = \exp\!\left(-g_t^{(b,h)} \frac{\Delta t_t^+}{\tau_b + \varepsilon}\right), \qquad \widetilde{M}_t^{(b,h)} = \alpha_t^{(b,h)} M_{t-1}^{(b,h)}$$
     Zero elapsed time ($\Delta t_t = 0$, e.g., intra-frame spatial video patches) yields exact identity retention $\alpha_t = 1$.
   - **Normalized queries/keys & Read-Before-Write surprise estimation:**
     $$q_t, k_t = \frac{u}{\sqrt{\|u\|_2^2 + 10^{-6}}} \quad (\|k_t\|_2 < 1 \text{ strictly everywhere, } C^\infty\text{ smooth})$$
     Prior readout $y_{t,\text{prior}}$ is retrieved from memory **strictly before** writing $t$. For exact tick-by-tick routing ($C = 1$), $y_{t,\text{prior}}^{(b,h)} = \widetilde{M}_t^{(b,h)} q_t^{(b,h)}$. For hardware-efficient **Causal Chunk-Anchor Surprise WY** ($C \in \{16, 32\}$), let $t_0 = C\lfloor t / C\rfloor$ be the causal chunk start; the anchor state $M_{\text{anc}} = M_{t_0 - 1}$ and cumulative decay $\bar{\alpha}_t = \prod_{j=t_0}^{t} \alpha_j$ produce:
     $$y_{t,\text{prior}}^{(b,h)} = \bar{\alpha}_t^{(b,h)} M_{\text{anc}}^{(b,h)} q_t^{(b,h)}, \qquad e_t = \sqrt{\frac{1}{d}\left\|\tilde{x}_t - \mathrm{SoftHeadNorm}(y_{t,\text{prior}}) W_{\text{prior}}\right\|_2^2 + 10^{-6}}$$
     *Limiting assumption & state cost:* Chunk-anchor surprise ($C > 1$) is strictly causal (zero future lookahead) and guarantees exact online-equals-offline equivalence between streaming single-tick inference and WY chunk-parallel training, at the explicit cost of storing $M_{\text{anc}}$, $\bar{\alpha}_t$, and chunk phase—approximately doubling matrix memory state bytes relative to $C=1$ while leaving the information-carrying matrix rank $H_{\text{tot}} d_v d_k$ unchanged.
   - **Width-invariant two-branch epistemic router:** Combines a width-scaling content branch on $\tilde{x}_t$ (`lr_mult = 1/m_d`) with a width-invariant metadata branch on $m_t = [\mathrm{Emb}(\text{mod}_t), \ln(1 + \Delta t_t^+), r^{\text{rel}}_t, \ln(1 + e_t)] \in \mathbb{R}^{19}$ (`lr_mult = 1.0` across all widths $128 \to 1920$ to prevent $\mathcal{O}(1/d)$ dilution of epistemic signals), outputting per-head write probabilities $p_t^{(b,h)} \in (0, 1)$ and per-bank explicit no-write probabilities $r_{\text{nw}, t}^{(b)} \in (0, 1)$:
     $$\eta_t^{(b,h)} = m^{\text{vld}}_t \, p_t^{(b,h)} \left(1 - r_{\text{nw}, t}^{(b)}\right) \in [0, 1)$$
   - **Normalized-key rank-1 delta overwrite & post-write readout (strict `float32`, spectral contraction guaranteed since $\eta_t \|k_t\|_2^2 < 1$):**
     $$M_t^{(b,h)} = \widetilde{M}_t^{(b,h)} + \eta_t^{(b,h)} \left(v_t^{(b,h)} - \widetilde{M}_t^{(b,h)} k_t^{(b,h)}\right) (k_t^{(b,h)})^\top, \qquad y_t = m^{\text{vld}}_t \, \mathrm{SoftHeadNorm}\!\left(M_t q_t; \varepsilon{=}10^{-2}, w_o^{\text{norm}}\right) W_o$$
3. **SwiGLU Feed-Forward Sublayer:** Standard gated MLP with exact multiple-of-16 FFN width solving (`_match_params_via_ffn_dim`) so structural comparators match trainable core parameter counts within $\le 0.4\%$.

**Matched Comparators & Ablations:**
- `slwm_v3_no_routing`: Identical parameters (`+0.000%`), $\tau_b$, $g_b$, FFN width, and aggregate write budget, replacing per-bank content-to-timescale specialization with uniform bank-averaged write allocation ($\bar{\eta}_t^{(h)} = \frac{1}{B}\sum_{b=1}^{B} \eta_{t,\text{raw}}^{(b,h)}$ broadcast identically across all $B$ banks; `models/v3/memory.py:460-461`).
- `single_bank_gated_deltanet`: Faithful $B=1$ single-timescale ($\tau^{(0)} = 64\text{ s}$) Gated DeltaNet with identical total matrix heads $H_{\text{tot}}$ and per-head $(d_k, d_v)$ rank, linear write gate $\beta_t = \sigma(\tilde{x}_t W_\beta + b_\beta)$, and no surprise/no-write router MLP.
- `sliding_window_gqa_eqbyte`: Pure sliding-window GQA removing the matrix memory sublayer, solving $W_{\text{eq}}$ to match total $C=1$ core streaming state bytes (`W_eq = 64` at `42.4M`, `128` at `137.8M`, `192` at `703.1M`) and matching trainable parameters via FFN width.

### 2.2 Wave B (`PolicyCommitGate`) & Wave C (Frozen-Core Decoders)

- **Wave B (`PolicyCommitGate`):** Operates strictly on `jax.lax.stop_gradient(Z_t)` and 5 frozen epistemic features `[norm_text_entropy, norm_audio_entropy, log1p(mean_prior_error), cumulative_mean_reliability, text_top1_prob]` (`scripts/train/v3_wave_b_policy_gate.py:142-146,234-266`) via a decoupled content MLP (`d -> 256 -> 5`) plus a width-invariant evidence branch with linear skip (`ev_skip: 5 -> 5`) and MLP (`5 -> 32 -> 5`), selecting among `{SPEAK_TEXT, EMIT_AUDIO, GENERATE_VIDEO, ACT_MOTOR, ABSTAIN_SILENCE}`. Parameter count is $257 d + 1{,}918$ (`0.199M` at `137.8M`, `0.397M` at `703.1M`, `0.495M` at `1.072B`).
- **Wave C (Frozen-Core Any-to-Any Decoders):** Verified bit-exact `max_core_grad == 0.0` into frozen core weights on all 15 runs (`weight_decay = 1e-4`). Augments linear readout heads with a residual 2-layer SwiGLU MLP branch (`h_dec = min(2048, max(512, 2d))`, zero-initialized output projection; `models/v3/any2any_heads.py:65-71`) plus a strictly causal intra-frame `3×5` spatial refiner (`2D 3×3` conv across the 15 Wan 2.1 patches of a single frame group with $\Delta t = 0$, zero temporal lookahead) trained with `0.5 * Huber + 0.5 * (Refined_Huber + 0.15 * SpatialGrad_L1)`.

### 2.3 Dual-Track Cost Accounting (`AGENTS.md §4.1`)

#### Track 1 — Core Comparison (Matched Trainable Cores & Streaming State Footprint)

| Scale Profile | Backbone Arm | $d$ / $L$ / $H_q$:$H_{\text{kv}}$ | $W$ ($\mathrm{RF}_{\text{loc}}$) | Mem $(B \times H_m, d_k)$ | Chunk $C$ | Trainable Core Params | Core State (`B=1`, MiB) |
|---|---|---|---:|---|---:|---:|---:|
| `discovery_45m` | `slwm_v3_routed_c16` / `c32` | `512 / 12 / 8:2` | `32` (`372`) | `4×1, 32` | `16` / `32` | `42,355,808` | `0.563`\* |
| `discovery_45m` | `slwm_v3_no_routing_c16` | `512 / 12 / 8:2` | `32` (`372`) | `4×1, 32` | `16` | `42,355,808` | `0.563`\* |
| `discovery_45m` | `single_bank_gated_deltanet` | `512 / 12 / 8:2` | `32` (`372`) | `1×4, 32` | WY | `42,500,396` | `0.375` |
| `discovery_45m` | `sliding_window_gqa_eqbyte` | `512 / 12 / 8:2` | `64` (`756`) | — | — | `42,249,920` | `0.376` |
| `confirmation_135m` | `slwm_v3_routed_c16` | `768 / 16 / 12:4` | `64` (`1,008`) | `4×1, 64` | `16` | `137,846,080` | `3.001`\* |
| `confirmation_135m` | `slwm_v3_no_routing_c16` | `768 / 16 / 12:4` | `64` (`1,008`) | `4×1, 64` | `16` | `137,846,080` | `3.001`\* |
| `confirmation_135m` | `single_bank_gated_deltanet` | `768 / 16 / 12:4` | `64` (`1,008`) | `1×4, 64` | WY | `137,554,512` | `2.001` |
| `confirmation_135m` | `sliding_window_gqa_eqbyte` | `768 / 16 / 12:4` | `128` (`2,032`) | — | — | `137,845,184` | `2.002` |
| `confirmation_135m` (`scratch`) | `slwm_v3_routed_c16` | `768 / 16 / 12:4` | `64` (`1,008`) | `4×1, 64` | `16` | `135,011,648`† | `3.001`\* |
| `confirmation_135m` (`scratch`) | `single_bank_gated_deltanet` | `768 / 16 / 12:4` | `64` (`1,008`) | `1×4, 64` | WY | `134,720,080`† | `2.001` |
| `scale_700m` | `slwm_v3_routed_c32` | `1536 / 24 / 24:4` | `64` (`1,512`) | `4×2, 64` | `32` | `703,149,536` | `7.502`\* |
| `scale_700m` | `single_bank_gated_deltanet` | `1536 / 24 / 24:4` | `64` (`1,512`) | `1×8, 64` | WY | `703,439,960` | `4.502` |
| `scale_700m` | `sliding_window_gqa_eqbyte` | `1536 / 24 / 24:4` | `192` (`4,584`) | — | — | `703,464,128` | `4.504` |
| `scale_1b` | `slwm_v3_routed_c32` | `1920 / 24 / 30:6` | `64` (`1,512`) | `4×2, 64` | `32` | `1,072,306,016` | `8.252`\* |
| `scale_1b` | `single_bank_gated_deltanet` | `1920 / 24 / 30:6` | `64` (`1,512`) | `1×8, 64` | WY | `1,072,677,080` | `5.252` |

*\*Note on State Accounting:* Base ($C=1$) matrix state + local GQA ring KV matches `single_bank` and `sliding_window_gqa_eqbyte` byte-for-byte (`0.375 MiB` at `42.4M`, `2.00 MiB` at `137.8M`, `4.50 MiB` at `703.1M`, `5.25 MiB` at `1.072B`). Enabling $C \in \{16, 32\}$ Chunk-Anchor Surprise WY adds the `M_anchor` + `cum_decay` buffer (`+0.188`, `+1.00`, `+3.00`, `+3.00 MiB`), reported explicitly above.  
*†`scratch_edge` omits the `32 × 2048` Mimi RVQ edge embedding tables (`-2.83M` params).*

#### Track 2 — Strict System Cost (All Inference-Active Trainable + Frozen Components)

| Component | Role / Mode | Status | Exact Parameters | Streaming State (`bf16` / `fp32`) | Causal $\mathcal{O}(1)$ Verified? | License & Upstream Provenance |
|---|---|---|---:|---:|---|---|
| **Kyutai Mimi** (`12.5 Hz`, `32 RVQ`, `4.4 kbps`) | Audio Encoder + RVQ | Frozen | `39,391,520` | `3.913` / `7.825 MiB` | Yes (`0.0` diff over `24 s`; KV cap `249`; zero leak `>79.7 s`) | `cc-by-4.0`; Moshi `7M`-hr speech/audio (public eval overlap **unknown/undisclosed**) |
| **Kyutai Mimi** | Audio Decoder + Upsample | Frozen | `39,917,089` | Included in Mimi codec | Yes (causal streaming decode) | Same as above (total Mimi: `79,308,609`) |
| **Wan 2.1 Causal VAE** (`192×320`, `25 fps`, `3×5` patches) | Video Encoder | Frozen | `53,596,928` | `164.531` / `329.062 MiB` | Yes (`0.0` prefix diff; `345.05 MB` exact constant `fp32` cache) | `apache-2.0`; open video VAE corpus (public eval overlap **unknown/undisclosed**) |
| **Wan 2.1 Causal VAE** | Video Decoder | Frozen | `73,295,603` | Causal 3D conv cache *(unmeasured)* | Yes (chunked causal decode) | Same as above (total Wan 2.1: `126,892,531`) |
| **ByT5 UTF-8 Byte Edge** (`264` vocab) | Text/Code Codec | Stateless | `0` | `0 B` | Yes (`100.0%` lossless, `0.0%` OOV) | Deterministic UTF-8 byte mapping |
| **Total Frozen Perceptual Edge** (`Enc` / `Enc+Dec`) | Shared across all arms | Frozen | `92,988,448` / `206,201,140` | `168.44` / `336.89 MiB` *(enc only)* | Verified (`gate2_codec_ceilings_receipt.json`) | Identical frozen codecs across all compared backbones |
| **Wave A Linear Heads** (`137.8M` / `703.1M` / `1.072B`) | Multimodal Next-Packet | Trainable | `13.62M` / `27.21M` / `34.01M` | `0 B` | Yes | Trained from scratch |
| **Wave B `PolicyCommitGate`** (`137.8M` / `703.1M` / `1.072B`) | Epistemic Commit Gate | Trainable | `0.199M` / `0.397M` / `0.495M` | `0 B` | Yes (`stop_gradient(Z_t)`) | Trained from scratch on frozen core |
| **Wave C SwiGLU + `3×5` Refiner** (`137.8M` / `703.1M` / `1.072B`) | Any-to-Any Decoders | Trainable | `51.36M` / `89.75M` / `102.84M` | `0 B` | Yes (`max_core_grad == 0.0`) | Trained from scratch on frozen core |
| **Strict Total Active Inference System (`Core + Wave B + Wave C + Frozen Enc+Dec`)** | **`137.8M` / `703.1M` / `1.072B` (`routed`)** | **Mixed** | **`395.6M` / `999.5M` / `1,381.8M`** | **`171.4` / `175.9` / `176.7 MiB`** *(Core + Frozen Enc `bf16`; Wan 2.1 dec cache unmeasured)* | **Strict $\mathcal{O}(1)$ causal end-to-end** | **Dominated in encoder state bytes by Wan 2.1 VAE (`164.5 MiB`), not core (`3.0–8.3 MiB`)** |

#### Hardware, Precision, Peak HBM & Measured Latency (`reports/v3/hw_bench/`)

All JAX (`0.11.x`) core runs used `bfloat16` matrix multiplications and local ring-KV caches with strict `float32` recurrent matrix states, RMSNorms, decay exponentials, and optimizer master weights (AdamW with global grad norm clipping `1.0`, `weight_decay = 0.01` on 2D core weights, `0.0` on scalar-role leaves, warmup + cosine decay, `base_lr = 2e-3` on multimodal `v2` runs and `3e-3` on `42.4M` RSO; Wave B/C decoders used `weight_decay = 1e-4`). Training executed on `NVIDIA A100-SXM4-80GB`, `NVIDIA L4`, `TPU v5e-1`, and `TPU v6e-1`; local frozen-weight long-horizon inference (`§6.3`) executed on `Apple M4 Max` (JAX CPU `float32` core + PyTorch `mps` Wan 2.1 / Mimi codec decode).

| Profile | Backbone | Params (`hw_bench`) | Chunk / `W` | `B × T` | Receipt Source (`A100-SXM4-80GB`) | Train Step (`ms`) | Throughput (`pos/s`) | `B=1` Stream Latency p50 / p95 (`ms`) | Peak HBM (`MiB`) |
|---|---|---:|---|---:|---|---:|---:|---:|---:|
| `confirmation_135m` | `slwm_v3_routed` | `137.85M` | `c=1` / `W=64` | `8 × 512` | `hw_bench_a100_wy_routed` | `1,523.3` | `2,688.8` | `9.29 / 11.08` | `4,231` |
| `confirmation_135m` | `slwm_v3_routed` | `137.85M` | `WY c=16` / `W=64` | `8 × 512` | `hw_bench_a100_wy_routed` | `336.5` (`4.53×`) | `12,173.6` | `8.79 / 9.32` | `5,269` |
| `confirmation_135m` | `slwm_v3_routed` | `137.85M` | `WY c=32` / `W=64` | `8 × 512` | `hw_bench_a100_wy_routed` | `246.5` (`6.18×`) | `16,615.2` | `11.57 / 12.10` | `5,271` |
| `confirmation_135m` | `single_bank_gated_deltanet`\* | `132.25M`\* | `WY` / `W=64` | `8 × 512` | `hw_bench_a100` (`c=1` ref `1,502.5 ms`) | `113.7` | `36,032.5` | `7.65 / 8.15` | `8,907` |
| `confirmation_135m` | `sliding_window_gqa`\* | `119.55M`\* | `W=64`\* (not `128`) | `8 × 512` | `hw_bench_a100` (`c=1` ref `1,502.5 ms`) | `60.4` | `67,865.4` | `5.17 / 5.29` | `8,907` |
| `scale_700m` | `slwm_v3_routed` | `703.15M` | `c=1` / `WY c=32` | `4 × 512` | `hw_bench_a100_wy_routed` | `2,422.2` / `414.6` (`5.84×`) | `845.5` / `4,940.3` | `17.57` / `18.67` (p50) | `15,163` / `16,146` |
| `scale_700m` | `single_bank_gated_deltanet`\* | `669.82M`\* | `WY` / `W=64` | `4 × 512` | `hw_bench_a100` (`c=1` ref `2,493.6 ms`) | `201.1` | `10,182.5` | `15.43 / 16.98` | `15,153` |
| `scale_1b` | `slwm_v3_routed` | `1,072.31M` | `c=1` / `WY c=32` | `2 × 512` | `hw_bench_a100_wy_routed` | `2,388.8` / `368.4` (`6.48×`) | `428.7` / `2,779.6` | `22.41` / `26.75` (p50) | `21,601` / `22,077` |

*\*Unmatched `hw_bench_a100.json` disclosure:* In the earlier `hw_bench_a100.json` benchmark pass, `--match-anchor` was disabled, so `single_bank` (`132.25M` at `135m`, `-4.1%`; `669.82M` at `700m`, `-4.7%`) and `sliding_window_gqa` (`119.55M`, `W=64` / `1.001 MiB`, not `W_eq=128`) had smaller FFN/window sizes than the matched training runs in Track 1, and `peak_bytes_in_use_mib` reflects a cumulative per-process HBM high-water mark across sequential suite rows (`8,907 MiB` carries over from `confirmation_135m B=32`). Speedups in parentheses (`4.53×`, `6.18×`, `5.84×`, `6.48×`) are strictly within-session ratios inside `hw_bench_a100_wy_routed.json`. Against those unmatched comparators at `seq_len = 512`, `slwm_v3_routed` is `2.96×` (`c=16`) / `2.17×` (`c=32`) slower per training step than `single_bank` at `137.8M`, `2.06×` slower at `703.1M` (`c=32`), and `5.57×` (`c=16`) / `4.08×` (`c=32`) slower than `W=64` GQA, while `B=1` streaming tick latency (`8.79 ms` at `137.8M`, `18.67 ms` at `703.1M`, `26.75 ms` at `1.072B`) stays well under the `80 ms` (`12.5 Hz`) Mimi audio packet period.

---

## 3. Experimental Methodology, Corpus Hygiene & `v1 → v2` Overfitting Ablation

### 3.1 Gate 1–2 Pre-Training Hardening & Corpus Inventory

Before scaling, the project passed independent Gate 1 (observability & control ceilings), Gate 2/2b/2c (codec causality & split disjointness), and Gate 3 (architecture, $\mu\text{P}$ coordinate check, and static zero-crutch audit):
- **Gate 1 Physical Observability & Oracle Ceilings (`gate1_control_receipt.json`, `gate1_holo_blind_scores.json`):** Synthetic rollout control with oracle perception & oracle dynamics reaches `87.5%` sustained tail reach across `1,024` worlds (`100%` on static wall, `50%` on dual chicane); blind 4-way visual observability auditing on HoloAssist by a single context-isolated blind AI subagent (`n = 20`, zero human raters) reaches `75.0%` overall (`90.0%` on clearly observable clips vs. `25%` chance).
- **Gate 2 Causal Codec Ceilings (`gate2_codec_ceilings_receipt.json`):** Mimi `32 RVQ` achieves `95.0%` Whisper-tiny Speech Commands ceiling (`72.5%` at `8 RVQ` vs. `97.5%` original PCM) and `0.9976` 20 ms onset correlation on GreatestHits; Wan 2.1 Causal VAE achieves `39.68 dB / 0.9814 SSIM` on physical xArm6 and `27.44 dB / 0.8873 SSIM` on HoloAssist (`192×320`, `25 fps`).
- **Gate 2 / 2b / 2c Split Disjointness & Zero-Contamination Audit:** Split integrity is verified across three committed receipts: `gate2_split_ledger_receipt.json` (`LibriSpeech 360` and `GreatestHits` date-disjoint splits), `gate2b_diverse_splits_receipt.json` (`Text & Code`, `Clotho`, `ESC-50`, `RSO L12/L16`, and `v1` AV/robotics, including `0/357` 64-byte matches against HumanEval/MBPP solutions and `0/500` against LAMBADA), and `gate2c_v2_expanded_splits_receipt.json` (non-vacuous exact SHA-256 episode-payload audit on all 5 expanded `v2` domain suites—`LibriSpeech 360`, `GreatestHits`, `MSR-VTT v2`, `VGGSound v2`, `Robotics v2`—verifying **zero** train-to-dev and **zero** train-to-test video, audio, or motor-action overlaps across every corpus, and `0` dev-to-test overlap on all domains except a single `1/200` visually distinct clip pair sharing an audio track between `VGGSound dev` and `VGGSound test` with `0` occurrences in train).

#### `v2` Multi-Domain Corpus Composition & Epoch-Capped Sampling (`temp_sqrt`, `max_ep <= 5.0`)

Each multimodal training batch samples across all 8 domains with probability proportional to $N_i^{0.35} w_i$, followed by iterative water-filling that caps expected domain passes at `max_epochs_per_domain = 5.0` (`scripts/train/v3_any2any_trainer.py:126-158`). Because smaller shards hit the `5.0`-epoch cap, water-filling assigns **`86.5%`** (`255.1M` bytes) of positions at `137.8M` and **`85.3%`** (`167.8M` bytes) at `703.1M`/`1.072B` to `Text & Code`. Multimodal batches apply `25%` random single-modality dropout (`data/v3_unified_shards.py:367-386`), zeroing the dropped modality's input, setting `target_valid_mask = 0.0` and `reliability = 0.0` on dropped ticks (suppressing supervision on zeroed targets) while keeping `valid_mask = 1.0` for dense execution, alongside multi-horizon latent prediction (`multi_horizon_weight = 0.15`, horizons `(1, 8, 32)` in `v3_any2any_trainer.py:394`) and `50%` episode-anchor transitions.

| Domain | Modalities | `v2` Train Ticks | Dev / Test Ticks | Disjointness Axis (Receipt) | Planned Epochs (`137.8M`, `294.9M` pos) | Planned Epochs (`703.1M`/`1.072B`, `196.6M` pos) |
|---|---|---:|---:|---|---:|---:|
| **Text & Code** (FineWeb-Edu + Stack Python) | Text/Code bytes | `540,003,662` | `9.01M` dev + 4 sealed benchmarks\* | Exact doc hash + 64B substring (`0` hits; Gate 2b) | `0.47` (`255.1M` B) | `0.31` (`167.8M` B) |
| **LibriSpeech** (`train-clean-360` / `dev-clean` / `test-clean`) | Audio + Text | `6,878,423` | `537,584` / `531,078` | Speaker & audio disjoint (`0` audio overlap; Gate 2 & 2c) | `2.68` | `1.77` |
| **Clotho** (environmental audio captions) | Audio + Text | `2,308,614` | `626,634` / `633,557` | Audio & caption disjoint (`0` overlap; Gate 2b) | `3.64` | `2.39` |
| **ESC-50** (50-class audio, Folds 1–4 / Fold 5) | Audio + Class Text | `129,664` | — / `32,416` | Official Fold-5 holdout (`0` overlap; Gate 2b) | `5.00` (cap) | `4.67` |
| **GreatestHits** (physical stick-impact AV) | Video + Audio | `248,694` (`219.8k` vid) | `116,745` / `165,072` | Recording-date & media disjoint (`0` overlap; Gate 2 & 2c) | `5.00` (cap) | `5.00` (cap) |
| **MSR-VTT** (open-domain video + clean caption; `0` audio ticks) | Video + Text | `969,776` (`870.0k` vid) | `123,895` / `124,087` | Video disjoint (`0` video overlap; Gate 2c) | `5.00` (cap) | `4.21` |
| **VGGSound** (synchronized 6 s AV events) | Video + Audio + Label | `1,035,295` (`870.0k` vid) | `134,309` / `133,524` | `0` train-eval video/audio overlap (`1/200` dev-test audio share, `0` video; Gate 2c) | `5.00` (cap) | `4.03` |
| **Robotics** (LeRobot ALOHA-Sim + PushT + LIBERO†) | Video + Action + Proprio | `213,855` (`64.0k` vid, `128.5k` act) | `22,294` / `22,878` | Trajectory & video disjoint (`0` media overlap; Gate 2c) | `5.00` (cap) | `5.00` (cap) |
| **Real-Speech Overwrite `L12_Weq64_additive`** | Speech Audio (`167.0 s` / `2,088` ticks) | `160` pairs (`146` spk) | `64` (`51` spk) / `64` (`53` spk) | Primary-speaker disjoint (`0` overlap; Gate 2b) | Dedicated `42.4M` benchmark (`§5.2`, dev eval) | — |
| **Real-Speech Overwrite `L16_Weq128_additive`** | Speech Audio (`274.0 s` / `3,425` ticks) | `192` pairs (`177` spk) | `96` (`68` spk) / `128` (`84` spk) | Primary-speaker disjoint (`0` overlap; Gate 2b) | `0-shot` sealed-test eval only (`§4.1`) | `0-shot` sealed-test eval only (`§4.1`) |

*\*Sealed text/code benchmarks: WikiText-103 test (`1.29M` bytes, `242.3k` words), LAMBADA OpenAI test (`5,153` passages), HumanEval (`164` problems), and MBPP Sanitized (`257` problems).*  
*†Within Robotics, ALOHA-Sim and PushT supply visual + motor streams, while LIBERO contributes proprioception, action, and text only (`0` video packets).*

### 3.2 `v1 → v2` Data Scaling & Overfitting Contrast (Non-Matched Exposure Confounds Disclosed)

Our initial `1.072B` fit-check (`v1`) stopped mid-cosine at `6,000 / 18,000` steps (`98.3M` total positions across a 7-domain mix without GreatestHits, using a `180M`-byte text shard, smaller AV/robotics shards without epoch capping that looped `14×–88×`, and a Python-list MSR-VTT caption formatting bug `['...']`). Because `v1` and `v2` differ simultaneously in total positions (`98.3M` vs. `196.6M–294.9M`), cosine completion, domain count (`7` vs. `8`), corpus size, and epoch capping, this is an **approximate multi-factor regime contrast**, not a single-factor ablation. Even with `7.8×` fewer parameters, `137.8M v2` cures VGGSound overfitting (`9.344 -> 6.008` CE) and decisively beats `1.072B v1` on text, code, ESC-50 probe, and MSR-VTT—while `1.072B v1` retained higher scores on Clotho F1, ESC-50 A2T, and Robotics CF gain from heavy multi-epoch passes on those smaller `v1` shards:

| Sealed-Test Metric | `1.072B v1` (`routed`, `98.3M` pos, mid-cosine, uncapped) | `137.8M v2` (`routed`, `294.9M` pos, `n=2`) | `703.1M v2` (`routed`, `196.6M` pos, `n=2`) | `1.072B v2` (`routed`, `196.6M` pos, `n=1`) |
|---|---:|---:|---:|---:|
| **WikiText-103 BPB ↓** (`approx PPL ↓`) | `2.317` (`5,166`) | `1.971` (`1,428`) | `1.945` (`1,293`) | **`1.935` (`1,260`)** |
| **LAMBADA Final-Word Acc ↑** (`BPB ↓`) | `0.27%` (`2.719`) | `4.65%` (`1.731`) | `5.38%` (`1.657`) | **`6.73%` (`1.603`)** |
| **HumanEval / MBPP Solution BPB ↓** | `1.450` / `1.680` | `1.110` / `1.265` | `1.075` / `1.226` | **`1.073` / `1.226`** |
| **ESC-50 Fold-5 Linear / MLP Top-1 ↑** | `51.75%` / `53.75%` | **`59.88%`** / `60.75%` | `59.38%` / **`61.00%`** | `57.00%` / `57.50%` |
| **ESC-50 `Audio -> Text` Shuffle Δ ↑** | **`+0.1296`** *(heavy looping)* | `+0.0499` | `+0.0935` | `+0.1029` |
| **Clotho Caption Token F1 ↑** (`causal Δ ↑`) | **`0.2142`** (`+0.0280`) | `0.1686` (`+0.0012`) | `0.1862` (`+0.0191`) | `0.1834` (`+0.0272`) |
| **VGGSound `Video -> Audio` Paired CE ↓** | `9.344` *(overfit collapse)* | **`6.008`** | `6.021` | `6.077` |
| **MSR-VTT Caption Token F1 ↑** (`causal Δ ↑`) | `0.1385` (`+0.0098`) | **`0.2123`** (`+0.0048`) | `0.1913` (`-0.0032`) | `0.1797` (**`+0.0183`**) |
| **LibriSpeech `Audio -> Text` Causal Δ ↑** (`WER ↓`) | `+0.0522` (`1.59`) | **`+0.0805`** (`1.56`) | `+0.0553` (`1.52`) | `+0.0742` (**`1.43`**) |
| **Robotics Action→Proprio / Vid CF Δ ↑** | **`6.833`** / **`0.0477`** *(heavy looping)* | `6.117` / `0.0100` | `5.860` / `0.0077` | `5.919` / `0.0070` |

---

## 4. Primary Sealed-Test Results Across Scales (`137.8M`, `703.1M`, `1.072B`)

All primary results below come strictly from frozen checkpoints evaluated on sealed test partitions (except Wave B and Wave C in `§6`, which evaluate on held-out dev partitions). Because `v2` confirmation and `703.1M` cells have `n=2` seeds (`s4201`, `s4202`), we report **`mean ± half-range`** (`|s4201 - s4202| / 2`, descriptive spread rather than a parametric 95% CI), and label single-seed cells (`n=1`, `s4201`) explicitly. **All zero-score cells, negative margins, and regressions appear directly in these primary tables (`AGENTS.md §13.1`).**

### 4.1 Language, Code & Long-Range Zero-Shot Recall (Sealed Test)

| Scale & Budget | Backbone Arm | Seeds | WikiText-103 BPB ↓ (`PPL ↓`) | LAMBADA Acc ↑ (`BPB ↓`) | HumanEval BPB ↓ (`Syn%` / `Pass@1`) | MBPP BPB ↓ (`Syn%` / `Pass@1`) | RSO L16 0-Shot Flip ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| **137.8M** (`294.9M` pos, `18k` steps) | `slwm_v3_routed_c16` | `2` | **`1.9706 ± 0.0036`** (`1,428`) | `4.65 ± 0.15%` (**`1.731`**) | `1.110 ± 0.001` (`42.1%` / **`0.0%`**) | **`1.265 ± 0.029`** (`42.4%` / **`0.0%`**) | **`0.0%`** |
| **137.8M** (`294.9M` pos) | `slwm_v3_no_routing_c16` | `2` | `1.9710 ± 0.0100` (`1,431`) | **`4.99 ± 0.02%`** (`1.734`) | **`1.081 ± 0.008`** (`22.9%` / **`0.0%`**) | `1.274 ± 0.012` (`9.3%` / **`0.0%`**) | **`0.0%`** |
| **137.8M** (`294.9M` pos) | `single_bank_gated_deltanet` | `2` | `2.0007 ± 0.0039` (`1,595`) | `3.49 ± 0.60%` (`1.820`) | `1.097 ± 0.018` (`24.1%` / **`0.0%`**) | `1.300 ± 0.022` (`3.1%` / **`0.0%`**) | **`0.0%`** |
| **137.8M** (`294.9M` pos) | `sliding_window_gqa_eqbyte` | `1`\* | `2.0320` (`1,790`) | `3.18%` (`2.466`) | `1.122` (`11.6%` / **`0.0%`**) | `1.290` (`79.4%` / **`0.0%`**) | **`0.0%`** |
| **703.1M** (`196.6M` pos, `12k` steps) | `slwm_v3_routed_c32` | `2` | `1.9449 ± 0.0048` (`1,293`) | `5.38 ± 0.18%` (`1.657`) | **`1.075 ± 0.003`** (`34.2%` / **`0.0%`**) | **`1.226 ± 0.013`** (`63.6%` / **`0.0%`**) | **`0.0%`** |
| **703.1M** (`196.6M` pos) | `single_bank_gated_deltanet` | `2` | **`1.9403 ± 0.0042`** (`1,271`) | `5.54 ± 0.49%` (**`1.650`**) | `1.077 ± 0.003` (`44.5%` / **`0.0%`**) | `1.250 ± 0.004` (`50.0%` / **`0.0%`**) | **`0.0%`** |
| **703.1M** (`196.6M` pos) | `sliding_window_gqa_eqbyte` | `2` | `1.9819 ± 0.0011` (`1,481`) | **`6.38 ± 0.38%`** (`2.055`) | `1.090 ± 0.038` (`28.4%` / **`0.0%`**) | `1.297 ± 0.054` (`11.3%` / **`0.0%`**) | **`0.0%`** |
| **1.072B** (`196.6M` pos, `12k` steps) | `slwm_v3_routed_c32` | `1`\* | **`1.9346`** (`1,260`) | **`6.73%`** (**`1.603`**) | **`1.073`** (`16.5%` / **`0.0%`**) | **`1.226`** (`7.8%` / **`0.0%`**) | **`0.0%`** |
| **1.072B** (`196.6M` pos) | `single_bank_gated_deltanet` | `1`\* | `1.9494` (`1,330`) | `5.55%` (`1.652`) | `1.080` (`14.6%` / **`0.0%`**) | `1.263` (`29.6%` / **`0.0%`**) | **`0.0%`** |

*\*Single-seed (`s4201`) cells: `confirmation_135m sliding_window_gqa_eqbyte` (`n=1`) and `scale_1b` (`n=1`, no `1.072B` GQA cell). Notice also the high seed variance in greedy Python syntax validity (`Syn%`, e.g., `81.7%` vs. `2.4%` across seeds on `137.8M routed` due to single-character indentation loops under greedy `T=0` decoding).*

### 4.2 Audio, Audio-Visual, Speech & Embodied Robotics Grounding (Sealed Test)

| Scale | Backbone Arm | Seeds | ESC-50 F5 Lin / MLP ↑ | ESC-50 A2T Δ ↑ | GreatestHits V2A Δ ↑ (Ver.) | VGGSound V2A CE ↓ (Δ ↑) | Clotho F1 ↑ (Δ ↑) | MSR-VTT F1 ↑ (Δ ↑) | LibriSpeech A2T Δ ↑ (`WER ↓`) | Robotics Proprio / Vid CF Δ ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **137.8M** | `slwm_v3_routed_c16` | `2` | `59.9` / `60.8%` | `+0.050 ± 0.011` | **`+0.0298 ± 0.0004`** (`2/2`) | `6.008` (`-0.0001`) | `0.169` (`+0.001`) | **`0.212`** (`+0.005`) | `+0.081` (`1.56`) | **`6.12`** / **`0.0100`** |
| **137.8M** | `slwm_v3_no_routing_c16` | `2` | `58.9` / `62.4%` | **`+0.064 ± 0.021`** | `+0.0296 ± 0.0053` (`2/2`) | **`6.007`** (`+0.0079`) | `0.177` (**`+0.014`**) | `0.209` (`+0.002`) | **`+0.086`** (**`1.38`**) | `6.08` / `0.0082` |
| **137.8M** | `single_bank_gated_deltanet` | `2` | `57.9` / `62.3%` | `+0.042 ± 0.005` | `+0.0263 ± 0.0035` (`2/2`) | `6.018` (**`+0.0081`**) | **`0.182`** (`+0.012`) | `0.203` (`-0.005`) | `+0.076` (`1.56`) | `5.91` / `0.0088` |
| **137.8M** | `sliding_window_gqa_eqbyte` | `1`\* | **`60.3`** / **`63.0%`** | `+0.041` | `+0.0177` (`1/1`) | `6.030` (`+0.0057`) | `0.176` (`+0.014`) | `0.203` (**`+0.014`**) | `+0.063` (`1.47`) | `5.86` / `0.0095` |
| **703.1M** | `slwm_v3_routed_c32` | `2` | `59.4` / **`61.0%`** | **`+0.094 ± 0.020`** | `+0.0301 ± 0.0061` (`2/2`) | **`6.021`** (`+0.0017`) | `0.186` (`+0.019`) | `0.191` (`-0.003`) | `+0.055` (**`1.52`**) | **`5.86`** / `0.0077` |
| **703.1M** | `single_bank_gated_deltanet` | `2` | `59.8` / `60.1%` | `+0.084 ± 0.008` | **`+0.0315 ± 0.0050`** (`2/2`) | `6.041` (**`+0.0023`**) | `0.185` (`+0.004`) | `0.181` (`+0.003`) | **`+0.076`** (`1.54`) | `5.78` / **`0.0101`** |
| **703.1M** | `sliding_window_gqa_eqbyte` | `2` | **`61.9`** / `59.5%` | `+0.049 ± 0.002` | `+0.0167 ± 0.0021` (`2/2`) | `6.045` (`+0.0017`) | **`0.189`** (**`+0.020`**) | **`0.193`** (**`+0.004`**) | `+0.063` (`1.53`) | `5.70` / `0.0091` |
| **1.072B** | `slwm_v3_routed_c32` | `1`\* | `57.0` / `57.5%` | **`+0.103`** | **`+0.0354`** (`1/1`) | **`6.077`** (`-0.0016`) | **`0.183`** (**`+0.027`**) | **`0.180`** (**`+0.018`**) | `+0.074` (**`1.43`**) | `5.92` / `0.0070` |
| **1.072B** | `single_bank_gated_deltanet` | `1`\* | **`59.5`** / **`58.0%`** | `+0.096` | `+0.0284` (`1/1`) | `6.093` (**`+0.0041`**) | `0.183` (`+0.007`) | `0.145` (`-0.001`) | **`+0.077`** (`1.69`) | **`6.25`** / **`0.0119`** |

**Key Structural Takeaways from Primary Tables 4.1 & 4.2:**
- **Scaling Under Fixed `196.6M` Positions (`703.1M → 1.072B`):** `slwm_v3_routed_c32` improves monotonically on WikiText-103 (`1.945 -> 1.935`), LAMBADA (`5.38% / 1.657 -> 6.73% / 1.603`), ESC-50 A2T (`+0.094 -> +0.103`), GreatestHits V2A (`+0.0301 -> +0.0354`), Clotho causal F1 margin (`+0.019 -> +0.027`), MSR-VTT causal F1 margin (`-0.003 -> +0.018`), and LibriSpeech WER (`1.52 -> 1.43`). By contrast, `single_bank_gated_deltanet` stalls or regresses from `703.1M` to `1.072B` on WikiText (`1.940 -> 1.949`), MBPP (`1.250 -> 1.263`), GreatestHits (`+0.0315 -> +0.0284`), MSR-VTT (`0.181 -> 0.145`), and LibriSpeech WER (`1.54 -> 1.69`). Because `1.072B` is `n=1`, this divergence is an unconfirmed hypothesis requiring `n >= 3` seeds and larger position budgets (`>= 1B` pos).
- **Qualitative LibriSpeech WER Failure Analysis:** Although teacher-forced `Audio -> Text` causal shuffle gains are consistently positive (`+0.055` to `+0.086` nats/byte across all 15 runs), greedy `T=0` autoregressive byte decoding collapses into the **exact same unconditioned phrase** across completely distinct utterances (e.g., on `137.8M routed s4201`, utterances `utt0000`, `utt0001`, and `utt0002` all decode identically to `"and the stranger was all the way to the stranger and the stranger was all the way..."`, yielding `WER = 1.61`, `CER = 1.19`). This reflects severe under-training for raw audio-to-byte alignment (`1.77–2.68` passes over `train-clean-360` without CTC alignment or repetition penalties).

---

## 5. Mandatory Single-Factor Ablations (`AGENTS.md §4.1 & §9`)

### 5.1 `2×2` `Backbone × Representation` Factorial (`137.8M`, `n=3` Seeds)

To satisfy `AGENTS.md §4.1` and isolate pretrained codec contributions from sequence-backbone effects, we ran a controlled `2×2` factorial at `137.8M` (`2,000` steps, `32.77M` positions, `n=3` seeds `s4201–s4203`) comparing `slwm_v3_routed_c16` vs. `single_bank_gated_deltanet` across `scratch_edge` (strictly causal 64-bin log-mel features, **0 frozen audio codec parameters**) and `frozen_codec_edge` (Kyutai Mimi 32-RVQ, `39.39M` frozen encoder parameters):

| Representation Edge | Backbone (`n=3`, `mean ± SD`) | Trainable Core | Frozen Audio Enc | ESC-50 F5 Linear ↑ | ESC-50 F5 MLP ↑ | ESC-50 A2T Δ ↑ | Clotho Caption F1 ↑ (Causal Δ ↑) |
|---|---|---:|---:|---:|---:|---:|---:|
| **`scratch_edge`** (causal 64-bin log-mel) | `slwm_v3_routed_c16` | `135.01M` | **`0`** | **`62.42 ± 1.66%`** | **`57.33 ± 1.38%`** | `+0.4021 ± 0.0046` | `0.0594 ± 0.0792` (`+0.0006`) |
| **`scratch_edge`** (causal 64-bin log-mel) | `single_bank_gated_deltanet` | `134.72M` | **`0`** | `61.42 ± 1.01%` | `56.25 ± 0.66%` | **`+0.4423 ± 0.0154`** | `0.0080 ± 0.0013` (`+0.0000`) |
| **`frozen_codec_edge`** (Kyutai Mimi 32-RVQ) | `slwm_v3_routed_c16` | `137.85M` | `39.39M` | `43.83 ± 2.50%` | `50.33 ± 1.89%` | `+0.2570 ± 0.0101` | **`0.2016 ± 0.0083`** (**`+0.0225`**) |
| **`frozen_codec_edge`** (Kyutai Mimi 32-RVQ) | `single_bank_gated_deltanet` | `137.55M` | `39.39M` | `43.67 ± 0.29%` | `49.17 ± 0.76%` | `+0.1967 ± 0.0370` | `0.1974 ± 0.0185` (`+0.0161`) |

*Disentangled Attribution & Caveat:* Representation accounts for a **`+18.6 pp` advantage for zero-pretrained `scratch_edge` log-mel** on fine-grained 50-class acoustic discrimination (`62.4%` vs. `43.8%`), proving that SLWM acoustic representation quality does not depend on Mimi pretraining. Conversely, `frozen_codec_edge` accounts for **`+0.14–0.19` Clotho caption F1** (`0.2016` vs. `0.0594`), showing that semantic audio-to-text alignment under low text exposure relies heavily on Mimi's speech/audio codebook structure. Within both representations, `slwm_v3_routed` modestly edges `single_bank` on ESC-50 linear/MLP and Clotho F1 (`+0.2` to `+1.2 pp`). **Transparent caveat:** These 2-corpus factorial runs looped `10.9×` on Clotho and `58.3–59.1×` on ESC-50 (`58.32×` frozen, `59.05×` scratch), so absolute scores should be interpreted as controlled factorial contrasts rather than epoch-capped generalization ceilings.

### 5.2 Long-Horizon Selective Memory: `167.0 s` Real-Speech Overwrite `L12_Weq64_additive` (`42.4M`, `n=3`, Dev Split)

To isolate beyond-receptive-field selective state overwrite without task opcodes, we evaluated 5 matched `42.4M` arms (`n=3` seeds each) on `Real-Speech Overwrite L12_Weq64_additive` (`2,088` ticks = `167.04 s` at `12.5 Hz`, `64` speaker-disjoint dev pairs = `38` overwrite / `26` catch across `51` unique primary speakers; final `757` ticks are bit-identical across counterfactual A/B pairs, strictly exceeding `discovery_45m` equal-byte GQA's `RF = 756`). Training used a 3-phase dynamic-clipbank curriculum totaling `1,050` steps (`250` Phase 0 word + `350` Phase A short + `450` Phase B long) at `base_lr = 3e-3`. **Note:** Results below are strictly on the speaker-disjoint **`dev` split** (`reports/v3/wave_a_rso_v3_45m/rso_v3_45m_dev_table.md`); sealed-test RSO evaluation under supervised training remains future work.

| Arm (`42.4M`, `L12_Weq64_additive` dev, `n=3`) | Core Params | State (`MiB`) | Mean `r_nowrite` | Inside-RF Diag ↑ | Overall Joint Acc ↑ (`mean ± SE` / `SD`) | Overwrite A/B Flip ↑ | Upd Flip ↑ | Preserved ↑ | Catch Dual ↑ | vs. `routed_c16` Δ (95% Cluster CI) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `single_bank_gated_deltanet` | `42.50M` | `0.375` | `0.000` | `21.35%` | **`16.15 ± 2.60%`** (`4.51%`) | **`12.28%`** | **`28.07%`** | **`57.02%`** | **`21.79%`** | `-2.6 pp [-12.5, +6.5]` |
| `slwm_v3_routed_c16` (anchor) | `42.36M` | `0.563` | `0.567` | **`21.88%`** | `13.54 ± 1.88%` (`3.25%`) | **`12.28%`** | **`28.07%`** | `52.63%` | `15.38%` | anchor |
| `slwm_v3_routed_c32` | `42.36M` | `0.563` | `0.616` | `19.27%` | `12.50 ± 4.13%` (`7.16%`) | `9.65%` | `22.81%` | `43.86%` | `16.67%` | `+1.0 pp [-9.1, +11.4]` |
| `slwm_v3_no_routing_c16` | `42.36M` | `0.563` | `0.741` | `17.71%` | `9.90 ± 2.27%` (`3.93%`) | `7.02%` | `23.68%` | `42.98%` | `14.10%` | `+3.6 pp [-3.8, +10.5]` |
| `sliding_window_gqa_eqbyte` | `42.25M` | `0.376` | `0.000` | `1.04%` | `2.08 ± 1.04%` (`1.80%`) | `0.00%` | `0.00%` | `21.93%` | `5.13%` | **`+11.5 pp [+4.1, +19.0]`** |

**Interpretation:** Equal-byte `Sliding-Window GQA` mathematically collapses to `0.00%` Overwrite A/B flip (`2.08%` joint) once the overwrite event leaves its `756`-tick receptive field, confirming a statistically significant `+11.5 pp` `[+4.1, +19.0]` selective-memory advantage for `slwm_v3_routed_c16`. Learned routing (`routed_c16`) outperforms content-independent uniform routing (`no_routing_c16`) by `+3.6 pp` overall and `+5.26 pp` on Overwrite A/B (`12.28%` vs. `7.02%`), while `single_bank_gated_deltanet` (`16.15%`) is statistically indistinguishable from `routed_c16` (`-2.6 pp [-12.5, +6.5]`) at `42.4M`. Notably, `slwm_v3_routed` suppresses writes on `56.7%` of distractor ticks (`mean_nowrite = 0.567`), whereas `single_bank` has no explicit write-abstention channel.

### 5.3 Scope Transparency on Uncompleted `AGENTS.md §9` Ablations

To maintain strict scientific disclosure, the following `AGENTS.md §9` ablations were **not** re-run on the `v2` epoch-capped `137.8M`–`1.072B` suite during this iteration and remain open for the next wave: `slwm_v3_no_nowrite`, `slwm_v3_no_zerolag`, `slwm_v3_tied_tau`, `diag_ssm`, `explicit_bounded_kv`, pure `text-only` vs. `signal-only -> text` pretraining, and context-length scaling beyond `T = 512` during multimodal core training (`T = 2,088` was exercised only on supervised `42.4M` RSO `L12_Weq64_additive`).

---

## 6. Wave B (`PolicyCommitGate`), Wave C (Frozen-Core Decoders) & `> 1.0 s` Long-Horizon Inference Audit

### 6.1 Wave B: Zero-Shortcut Epistemic Commitment (`stop_gradient(Z_t)`, Dev Split)

Per `AGENTS.md §8`, we never report unsupported-claim reduction without simultaneously reporting overall decision accuracy, grounded content accuracy on committed outputs (usefulness), contradiction rate, abstention rate, and Expected Calibration Error (`ECE`). In the **primary zero-shortcut blind regime** (`n = 2,048` dev samples), `reliability == 1.0` identically on both supported and silently mismatched unsupported episodes, forcing `PolicyCommitGate` to rely strictly on frozen core $Z_t$ and the 5 epistemic features (`always_commit_baseline` suffers `100.0%` unsupported claims with per-arm grounded content accuracy spanning `66.70%–69.34%` [`65.72%–69.43%` per seed]; `reliability_only_threshold_baseline` scores `47.9%–49.9%` decision accuracy; gate pass predicate in `v3_wave_b_policy_gate.py:490` is strict `> +5.0 pp`).

| Scale | Backbone Arm | Seeds | Blind Gate Pass (`> +5 pp`) | Blind Dec. Acc ↑ | Gain vs. Rel-Only ↑ | Committed Content Acc ↑ (Always-Commit) | Blind Unsup. Rate ↓ (`Red. vs. Always`) | Contradiction Rate ↓ | Abstention Rate | **Blind ECE ↓** | Degraded Outage Acc ↑ / Unsup ↓ / ECE ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **137.8M** | `slwm_v3_routed_c16` | `2` | **`2/2`** | `53.64%` | **`+5.35 ± 0.22 pp`** | `71.97%` (`67.14%`) | `36.77%` (`-63.2 pp`) | `6.18%` | `59.57%` | **`0.397`** | `68.98%` / `10.21%` / `0.264` |
| **137.8M** | `slwm_v3_no_routing_c16` | `2` | `1/2` | `52.81%` | `+4.40 ± 0.98 pp` | `72.63%` (`67.62%`) | **`35.26%`** (`-64.7 pp`) | **`5.64%`** | `61.87%` | **`0.407`** | `67.96%` / `8.84%` / `0.279` |
| **137.8M** | `single_bank_gated_deltanet` | `2` | `0/2` | `53.32%` | `+3.81 ± 0.59 pp` | `69.16%` (`67.53%`) | `41.22%` (`-58.8 pp`) | `7.45%` | `55.40%` | **`0.397`** | `69.10%` / `12.46%` / `0.262` |
| **137.8M** | `sliding_window_gqa_eqbyte` | `1`\* | `0/1` | `52.39%` | `+2.64 pp` | `71.59%` (`67.97%`) | `38.77%` (`-61.2 pp`) | `6.25%` | `58.79%` | **`0.416`** | `69.60%` / **`5.96%`** / `0.265` |
| **703.1M** | `slwm_v3_routed_c32` | `2` | **`1/2`** | **`54.45%`** | **`+4.91 ± 0.32 pp`** | `71.95%` (`67.77%`) | **`37.41%`** (`-62.6 pp`) | `6.60%` | `58.03%` | **`0.385`** | `69.62%` / `7.19%` / `0.255` |
| **703.1M** | `single_bank_gated_deltanet` | `2` | `0/2` | `54.08%` | `+4.69 ± 0.15 pp` | `72.38%` (`67.72%`) | `40.53%` (`-59.5 pp`) | `6.72%` | `55.40%` | **`0.396`** | `73.23%` / `6.50%` / `0.226` |
| **703.1M** | `sliding_window_gqa_eqbyte` | `2` | `0/2` | `53.88%` | `+4.06 ± 0.54 pp` | `71.13%` (`66.70%`) | `39.75%` (`-60.3 pp`) | `6.84%` | `56.35%` | **`0.387`** | `72.96%` / **`4.60%`** / `0.225` |
| **1.072B** | `slwm_v3_routed_c32` | `1`\* | `0/1` | `52.29%` | `+2.98 pp` | **`75.80%`** (`69.34%`) | `38.09%` (`-61.9 pp`) | **`5.18%`** | `59.57%` | **`0.413`** | `71.16%` / **`4.40%`** / `0.248` |
| **1.072B** | `single_bank_gated_deltanet` | `1`\* | `0/1` | `52.44%` | `+3.71 pp` | `72.35%` (`69.34%`) | `44.73%` (`-55.3 pp`) | `6.93%` | `52.73%` | **`0.411`** | **`73.90%`** / `6.84%` / **`0.224`** |

**Honest Assessment:** `slwm_v3_routed` achieves the highest blind gain over `reliability_only` at both `137.8M` (`+5.35 pp`, `2/2` pass) and `703.1M` (`+4.91 pp`, `1/2` pass) while maintaining `71.95%–75.80%` committed content accuracy (`+4.18` to `+6.46 pp` above each arm's own always-commit baseline) and non-collapsing abstention (`58.0%–59.6%`). However, **three serious weaknesses remain**: (1) absolute blind decision accuracy plateaus near `52.3%–54.5%` (modest above `~49%` reliability-only), (2) `ECE` sits at `0.385–0.416` blind (`0.224–0.279` degraded), demonstrating severe overconfidence in uncalibrated gate logits, and (3) `EMIT_AUDIO` and `ACT_MOTOR` are `0.0%` across all 15 runs (`GENERATE_VIDEO <= 0.49%`) because the current Wave B harness excludes robotics and masks evaluation to post-text transitions.

### 6.2 Wave C: Frozen-Core Decoders & Causal Conditioning Proofs (`max_core_grad == 0.0`, Dev Split)

Training decoupled residual-SwiGLU decoders + causal `3×5` spatial video refiner (`1,500` steps) strictly on `stop_gradient(Z_t)` (`max_core_grad == 0.0` verified on all 15 runs) tests causal cross-modal and action conditioning on held-out dev episodes without degrading core checkpoints:

| Scale | Backbone Arm | Seeds | Wave C Decoder Params | Robotics Next-Video Huber ↓ | Robotics Action→Vid CF Δ ↑ | Robotics Next-Proprio Huber ↓ | Robotics Action→Prop CF Δ ↑ | GreatestHits V2A Shuffle Δ ↑ (Ver.) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **137.8M** | `slwm_v3_routed_c16` | `2` | `51.36M` | `0.1040 ± 0.0018` | `+0.0180 ± 0.0006` | `0.7347 ± 0.0338` | **`+8.07`** | `+0.0071 ± 0.0050` (`0/2`) |
| **137.8M** | `slwm_v3_no_routing_c16` | `2` | `51.36M` | **`0.1035 ± 0.0048`** | `+0.0180 ± 0.0007` | `0.7029 ± 0.0336` | `+7.85` | `+0.0085 ± 0.0071` (`0/2`) |
| **137.8M** | `single_bank_gated_deltanet` | `2` | `51.36M` | `0.1145 ± 0.0008` | `+0.0177 ± 0.0015` | **`0.6892 ± 0.0231`** | `+7.87` | `+0.0104 ± 0.0007` (`0/2`) |
| **137.8M** | `sliding_window_gqa_eqbyte` | `1`\* | `51.36M` | `0.1097` | `+0.0180` | `0.8585` | `+7.42` | `+0.0126` (`0/1`) |
| **703.1M** | `slwm_v3_routed_c32` | `2` | `89.75M` | **`0.1263 ± 0.0026`** | `+0.0239 ± 0.0007` | `0.8773 ± 0.0308` | `+7.51` | `+0.0170 ± 0.0006` (`0/2`) |
| **703.1M** | `single_bank_gated_deltanet` | `2` | `89.75M` | `0.1402 ± 0.0014` | **`+0.0250 ± 0.0012`** | **`0.8487 ± 0.0249`** | `+7.73` | `+0.0170 ± 0.0047` (`1/2`) |
| **703.1M** | `sliding_window_gqa_eqbyte` | `2` | `89.75M` | `0.1400 ± 0.0011` | `+0.0231 ± 0.0008` | `1.0125 ± 0.0226` | `+7.51` | `+0.0029 ± 0.0028` (`0/2`) |
| **1.072B** | `slwm_v3_routed_c32` | `1`\* | `102.84M` | **`0.1600`** *(under-exposed)* | `+0.0244` | `1.0376` | `+7.50` | **`+0.0208`** (`1/1`) |
| **1.072B** | `single_bank_gated_deltanet` | `1`\* | `102.84M` | `0.1606` *(under-exposed)* | **`+0.0276`** | **`0.9266`** | **`+8.26`** | `+0.0162` (`1/1`) |

**Root Cause of Scale Degradation on Robotics Video Huber (`0.104 → 0.126 → 0.160`):** Because `max_epochs = 5.0` caps total robotics exposure at `5.0 × 213,855 = 1.07M` ticks across all scales, robotics exposure per parameter drops `7.8×` from `0.0078` (`137.8M`) to `0.0015` (`703.1M`) and `0.0010` (`1.072B`)—starving larger models of embodied visual dynamics even while their action-intervention counterfactual separation rises (`+0.018 -> +0.024`). Meanwhile, `SLWM-v3` scales cleanly on GreatestHits `Video -> Audio` Wave C shuffle gain (`+0.0071 -> +0.0170 -> +0.0208`, reaching verified significance at `1.072B`).

### 6.3 Long-Horizon (`> 1.0 s`) Frozen-Weight Local Inference & Artifact Audit (`AGENTS.md §13.2`)

To audit free-running autoregressive behavior beyond single-step teacher forcing, we executed `scripts/eval/v3_long_horizon_eval_and_review.py` with strictly frozen weights (`s4201` checkpoints + Wave C SwiGLU/Refiner heads) across all five converged `v2` models on fixed sealed-test episodes (`reports/v3/v2_long_horizon_frozen_eval_receipt.json`), computing objective DSP/vision/counterfactual metrics with explicit epistemic tags (`observed_ground_truth`, `predicted_free_running_frozen_weights`, `counterfactual_*`, `imagined_*`):

| Model (`s4201`, Frozen Weights) | `action2video` (`1.80 s` / `45f`) PSNR ↑ (`CF Δ`, Drift ↓, Huber ↓) | `video2action` (`2.40 s` / `60` steps, open-loop blind-action) Huber ↓ (`Cos ↑`) | `video2audio` (`4.0 s` GH) Mel-L1 ↓ (`env_r`, RVQ0 Rep ↓) | `audio2audio` (`6.0 s` Clotho) Mel-Cos ↑ (`RVQ0 Rep ↓`) | `text2audio` (`4.0 s`, 3 prompts) Centroid Spread ↑ (`Hz`) | `audio2video` (`2.96 s` / `37f`) PSNR ↑ (`Lat. Cos ↑`) |
|---|---:|---:|---:|---:|---:|---:|
| **`135m_routed`** | **`20.47 dB`** (`-0.24 dB`, `1.27×`, `0.159`) | **`0.053`** (`0.774`) | `2.12` (`+0.046`, **`91.8%`**) | `0.9995` (**`100.0%`**) | `1,431 Hz` | `12.33 dB` (`0.582`) |
| **`700m_routed`** | `18.60 dB` (**`+4.10 dB`**, `1.41×`, `0.220`) | `0.170` (`0.496`) | `3.62` (`+0.076`, **`91.8%`**) | `0.9994` (**`93.9%`**) | `2,508 Hz` | `12.36 dB` (`0.389`) |
| **`700m_single_bank`** | `19.71 dB` (**`+5.60 dB`**, `1.31×`, `0.244`) | `0.143` (`0.580`) | **`1.04`** (`+0.009`, `46.9%`) | `0.9995` (**`100.0%`**) | `1,082 Hz` | `12.10 dB` (`0.362`) |
| **`700m_sw_gqa`** | `19.33 dB` (`+2.89 dB`, **`1.10×`**, `0.214`) | `0.101` (**`0.778`**) | `1.88` (`+0.010`, `79.6%`) | **`0.9999`** (**`98.0%`**) | `732 Hz` | `12.09 dB` (`0.429`) |
| **`1b_routed`** | **`10.44 dB`** *(`-0.91 dB`, `1.85×`, `1.402`, collapse)* | `0.163` (`0.345`) | `3.30` (`-0.110`, **`89.8%`**) | `0.9994` (**`100.0%`**) | **`4,020 Hz`** | **`12.68 dB`** (`0.479`) |

**Critical Peer-Review Audit of Generated `> 1.0 s` Artifacts & `§13.2` Showcase Gaps:**
1. **`action2video` (`1.80 s` / `45` frames, ALOHA-Sim twin-arm manipulation):** Fixing shard-exact action-video interleaving and removing future proprioception leakage in `scripts/eval/v3_live_any2any_inference.py` brought 12-group (`45`-frame) free-running `135m_routed` rollout Huber to `0.159` (`20.47 dB`, `1.27×` late/early drift) with zero `3×5` tile seams (validating the causal spatial refiner), though with `-0.24 dB` counterfactual margin due to static copy-forward inertia. At `703.1M`, all three models preserve coherent twin-arm structure over `1.80 s` (`18.60–19.71 dB`, `1.10–1.41×` drift) and exhibit strong causal action sensitivity (`+2.89` to `+5.60 dB`), though fine gripper tips and peg contacts blur and negated counterfactual actions (`-a`) partially push rollouts off-manifold into Wan 2.1 checkerboard noise. At `1.072B`, robotics under-exposure (`0.001` ticks/param) causes geometric collapse (`10.44 dB`, `4.52` RGB motion L1 vs. `1.76` GT).
2. **`text2audio` (`4.0 s`), `audio2audio` (`6.0 s`) & `video2audio` (`4.0 s`):** Three distinct text prompts (`"water rushing in a stream"`, `"dog barking loudly"`, `"drum beats and percussion"`) shift spectral centroid strongly in `slwm_v3_routed` (`2,508 Hz` spread at `703.1M`, `4,020 Hz` at `1.072B` vs. `732 Hz` for `700m_sw_gqa`), **but** this reflects stationary timbre shifts rather than dynamic sound events: `text2audio` exhibits high temporal stationarity (`lag1_mel_cos = 0.9989–0.9998`), `34.7%–89.8%` RVQ-0 adjacent repetition, and reversed prompt-frequency ordering between `700m_routed` (`water 668 Hz < dog 1,009 Hz < drums 3,176 Hz`) and `1b_routed` (`water 4,945 Hz > drums 3,665 Hz > dog 926 Hz`). Objective RVQ-0 telemetry exposes **severe autoregressive RVQ-0 repetition collapse** on `audio2audio` (`93.9%–100.0%` adjacent repeats, `0.0–1.05` bits entropy vs. `3.3–3.8` bits GT) and `46.9%–91.8%` on `video2audio` (`89.8%–91.8%` on `slwm_v3_routed`; `46.9%` on `700m_single_bank` GreatestHits; `49.0%` on `700m_sw_gqa` VGGSound). Furthermore, GreatestHits `video2audio` runs open-loop after a video prefix without frame-interleaved visual clock ticks, yielding near-zero impact envelope correlation (`env_r = -0.11..+0.08`) despite `96–100%` counterfactual code divergence.
3. **`audio2video` & `text2video` (`2.96 s` / `37` frames):** Deterministic Huber regression in Wan 2.1 latent space averages over multimodal visual uncertainty, collapsing under-determined `audio2video` and `text2video` rollouts to smooth chromatic fog (`12.1–12.7 dB` PSNR). This cleanly separates well-determined dynamics (`action2video`, where deterministic Huber suffices for `18.6–20.5 dB` structure) from stochastic visual generation (requiring a flow-matching/diffusion head on frozen $Z_t$).
4. **`text/code` (`256 B`) & `video2text` Evaluation Bug:** Free-running `256`-byte greedy text/code rollouts loop quickly (`"was a star and the star was a star..."`). On MSR-VTT `video2text`, `scripts/eval/v3_live_any2any_inference.py#L406` truncates visual prefixes at `min(ctx_end, 512)`, cutting clips exceeding `512` ticks before the caption separator and inducing unconditioned prior-driven prose.
5. **Disclosed `AGENTS.md §13.2` Showcase Gaps:** While 135 tracked deterministic `inference_showcase/` manifests across `v2_*` implement frozen-weight `P05/P25/P50/P75/P95` quantile selection and explicit epistemic provenance labels, three `§13.2` items remain incomplete for the next iteration: (a) quantile episode IDs were ranked independently per model rather than locked to a shared cross-model episode manifest, (b) no trivial copy/retrieval baseline is rendered side-by-side in the manifests or Table 6.3 evaluation, and (c) committed `inference_showcase/` manifests cover text/captioning and Wave B decisions, whereas non-text media audits (`Table 6.3`) evaluate fixed sealed-test episodes (`ep0`/`ep1`) rather than committed multi-episode quantile galleries.

---

## 7. External Published Baseline Context (Approximate — Non-Matched Tokenization & Exposure)

Per `AGENTS.md §4` and `§10`, comparisons against external published models below are explicitly labeled **approximate** because external models use subword BPE tokenizers, text-only objectives, and vastly larger token budgets (`~4 B/tok` vs. `SLWM-v3`'s `167.8M–255.1M` [`0.17–0.26B`] text/code bytes within a `196.6M–294.9M` 8-domain multimodal budget):

| Reference System | Params | Tokenizer / Modality | Text Exposure (Approx. Bytes at `4 B/tok`) | Published LM PPL ↓ (Metric Scope) | LAMBADA Acc ↑ | HumanEval Pass@1 ↑ | Approx. Exposure Ratio vs. `SLWM-v3` Text Bytes |
|---|---:|---|---:|---:|---:|---:|---:|
| **GPT-2 Small** (Radford et al., 2019) | `124M` | 50k BPE (text-only) | `~10B` tok (`~40B` B) | `37.5` (WikiText-103) | `45.9%` | — | `~160×–240×` |
| **Pythia-160M / 1B** (Biderman et al., 2023) | `160M` / `1.0B` | 50k BPE (Pile text+code) | `300B` tok (`~1,200B` B) | `~29` / `~14` (WikiText) | `32.8%` / `56.1%` | `<1%` / `~3%` | `~4,700×–7,150×` |
| **Mamba-130M / 790M** (Gu & Dao, 2023) | `130M` / `790M` | 50k BPE (Pile text+code) | `300B` tok (`~1,200B` B) | `~20` / `~11` (Pile/WT-2 approx.) | `44.3%` / `62.7%` | — | `~4,700×–7,150×` |
| **Gated DeltaNet-340M / 1.3B** (Yang et al., 2024) | `340M` / `1.3B` | subword (FineWeb-Edu) | `15B–100B` tok (`~60B–400B` B) | `~21` / `~16` (WikiText-2/eval approx.) | `~45%` / `~60%` | — | `~240×–2,400×` |
| **`SLWM-v3` (`137.8M` / `703.1M` / `1.072B` `v2`)** | `138M`–`1.07B` | **UTF-8 bytes + 4 signal modalities** | **`0.17–0.26B` text bytes** (`0.20–0.29B` total pos) | `1,428` / `1,293` / `1,260` (WikiText-103 byte-to-word PPL; `1.97`–`1.93` BPB) | `4.7%` / `5.4%` / `6.7%` | `0.0%` | `1.0×` (Strict matched baselines in `§4` & `§5`) |

---

## 8. Comprehensive Limitations, Failure Modes & Next-Iteration Roadmap

1. **Severe Under-Exposure Relative to Parameter Scale (`0.18–2.1` pos/param overall; `0.001` robotics pos/param at `1.072B`):** Chinchilla-optimal text scaling requires `~20` tokens/param (`~80` bytes/param). Our `v2` runs operate at `2.1` (`137.8M`), `0.28` (`703.1M`), and `0.18` (`1.072B`) total multimodal positions/param. This explains `HumanEval/MBPP pass@1 = 0.0%`, LibriSpeech greedy WER (`1.38–1.69`), and `1.072B` robotics collapse. **Fix:** Expand robotics & AV corpora `10×` (Droid/OpenX-Embodiment, full VGGSound/AudioSet) and scale position budgets proportionally with model width (`>= 2B–5B` positions for `700M–1B`).
2. **Statistical Power Gaps (`n=1` & `n=2` Cells; Dev-Only RSO):** `confirmation_135m` and `scale_700m` have `n=2` seeds (`sw_gqa 135m` is `n=1`), `scale_1b` is `n=1` without a `1.072B` GQA arm, and supervised `42.4M` RSO `L12_Weq64_additive` was evaluated on the `dev` split only. **Fix:** Complete `n=3` seeds across `137.8M`/`703.1M`, add `1.072B sliding_window_gqa_eqbyte`, and run sealed-test evaluation on both `Real-Speech Overwrite L12_Weq64_additive` and `L16_Weq128_additive` test partitions with hierarchical seed+speaker cluster CIs.
3. **Chunk-Anchor State Overhead (`M_anchor`) & Throughput Gap vs. `Single-Bank DeltaNet`:** `slwm_v3_routed` uses `1.5×–1.67×` larger core state (`3.0` vs. `2.0 MiB` at `137.8M`; `7.5` vs. `4.5 MiB` at `703.1M`) and `~2.1×–3.0×` longer training steps than `single_bank_gated_deltanet`, while `single_bank` remains competitive at `703.1M` and `42.4M` RSO. **Fix:** Investigate fused Pallas/Triton chunked DeltaNet kernels and lightweight scalar-summary surprise estimators that eliminate the `M_anchor` copy while preserving exact online-offline equivalence.
4. **Autoregressive RVQ-0 Collapse & Open-Loop `video2audio` Synchronization:** Free-running audio generation collapses to `94–100%` (`audio2audio`) and `47–92%` (`video2audio`) RVQ-0 code repetition, and GreatestHits `video2audio` lacks frame-interleaved visual clocks (`env_r ≈ 0`). **Fix:** Implement strictly causal interleaved `video_group(t) -> 2 audio_ticks(t)` inference scheduling, add nucleus/min-p sampling with RVQ-0 repetition penalty, and evaluate causal multi-codebook delay patterns (`Moshi`/`MusicGen`).
5. **Deterministic Huber Video Blur (`audio2video` / `text2video`):** L1/Huber regression averages over uncertain future visual modes (`12.1–12.7 dB` fog). **Fix:** Train a causal conditional flow-matching (rectified flow) Wave C video head strictly on `stop_gradient(Z_t)` while keeping world-core weights frozen.
6. **Wave B Calibration (`ECE = 0.38–0.42`), Missing Action Modalities, `video2text` Bug & `§13.2` Showcase Gaps:** Add post-hoc temperature scaling + Brier/focal calibration to `PolicyCommitGate`, exercise all 5 commitment channels (`EMIT_AUDIO`, `GENERATE_VIDEO`, `ACT_MOTOR`), fix `min(ctx_end, 512)` in `scripts/eval/v3_live_any2any_inference.py#L406` to retain the final `512` visual ticks plus separator, measure the Wan 2.1 decoder causal conv cache in Track 2, and close the three `§13.2` showcase gaps (pre-locked common cross-model episode manifest, side-by-side trivial copy/retrieval baseline, and committed non-text quantile galleries).

---

## 9. Reproducibility Manifest & Tracked Provenance

All quantitative claims in this report are reproducible from committed repository code, frozen receipts, and deterministic inference showcase manifests on branch `research/slwm-v3-selective-memory`:

- **Core Architecture & Specifications:** `models/v3/spec.py`, `models/v3/memory.py`, `models/v3/core.py`, `models/v3/fast_core.py`, `models/v3/any2any_heads.py`
- **Training & Evaluation Pipelines:** `scripts/train/v3_any2any_trainer.py`, `scripts/train/v3_rso_benchmark_runner.py`, `scripts/train/v3_wave_b_policy_gate.py`, `scripts/train/v3_wave_c_decoders.py`, `scripts/eval/v3_live_any2any_inference.py`, `scripts/eval/v3_long_horizon_eval_and_review.py`
- **Gate 1–3 Hardening Receipts:** `reports/v3/gate1_control_receipt.json`, `reports/v3/gate1_holo_blind_scores.json`, `reports/v3/gate2_codec_ceilings_receipt.json`, `reports/v3/gate2_split_ledger_receipt.json`, `reports/v3/gate2b_diverse_splits_receipt.json`, `reports/v3/gate2c_v2_expanded_splits_receipt.json`, `reports/v3/gate3_architecture_and_mup_receipt.json`
- **Hardware Benchmarks:** `reports/v3/hw_bench/hw_bench_a100.json`, `reports/v3/hw_bench/hw_bench_a100_wy_routed.json`, `reports/v3/hw_bench/hw_bench_tpu_v6e1.json`
- **Experimental Receipts & Tracked `inference_showcase/` Manifests:**
  - `42.4M` RSO `L12_Weq64_additive` (`15` runs, `n=3`, dev split): `reports/v3/wave_a_rso_v3_45m/` (`rso_v3_45m_dev_table.md` + 15 receipts)
  - `137.8M` `2×2` `Backbone × Representation` (`12` runs, `n=3`, sealed test): `reports/v3/ablation_2x2_scratch_135m/`, `reports/v3/ablation_2x2_frozen_135m/`
  - `137.8M v2` Confirmation (`7` core + `7` Wave B + `7` Wave C + dev/test showcases): `reports/v3/v2_confirmation_135m_300m/`
  - `703.1M v2` Scale-Up (`6` core + `6` Wave B + `6` Wave C + dev/test showcases): `reports/v3/v2_scale_700m_200m/`
  - `1.072B v2` & `v1` Contrast (`2` `v2` core + `2` Wave B + `2` Wave C + `1` `v1` baseline + dev/test showcases): `reports/v3/v2_scale_1b_200m/`, `reports/v3/wave_d_scale_1b_300m/`
  - `> 1.0 s` Frozen-Weight Long-Horizon Audit (`5` converged `v2` checkpoints, sealed test): `reports/v3/v2_long_horizon_frozen_eval_receipt.json`
