# Forty multimodal training runs on one Colab TPU

**9 November 2026.** The strongest intervention was to train with a timing error the model had to detect. With a dense Transformer this improved held-out video-to-audio prediction in both seeds. The generated speech still failed a word-content test. Scaling the signal core, or tying every decoder to the same latent coordinates, did not produce a reliably interchangeable audiovisual representation.

## What was trained

All 40 G4 endpoints start from random weights. A single observed modality passes through one shared causal field. The output heads predict audio features, visual intensities, text or recorded robot quantities. GPT-2-style controls use dense causal attention and dense feed-forward layers. The signal core uses gated multiscale causal filters. Its sparse variant executes a grouped feed-forward matrix and retains full-width mixing between groups. Each family uses the same edges, heads, train order and effective batch within its comparison. The earlier 500M-token pretrained G3 models remain a separate study. [Registered protocol](../../../../docs/research/crossmodal_state_tpu_20261109.md)

The training pack contained **95,177 real examples and 6,088,567 valid positions**. It included 12,704 GRID clips, 16,384 LibriSpeech windows, 32,768 captioned COCO images, 1,440 Clotho clips, 7,305 DROID windows and 24,576 FineWeb text windows. AMI contributed zero training examples. In a 256-update full-data run the actual visible encoder views were **20.75% audio, 24.48% visual, 6.98% numeric and 47.79% text**. That run visited 54,217 distinct packed examples. The 1,024-update run visited all 95,177. Examples can share a recording or scene, and image positions are spatial raster patches. The 40 runs together account for 3,145,728 example presentations. [Exposure counts](exposure-counts.json) · [Data and endpoint identities](data_and_endpoints.json)

## Batching and speed

The sweep used actual forward passes, gradients and Adam updates, two observed views per example, 64 positions and an effective batch of 256. Arrays were verified and loaded into accelerator memory once. Timed blocks synchronized before returning. The table uses the fastest tested microbatch on each device. These measurements include sample-index generation and dispatch. Initial data loading, compilation, checkpoint writes and full validation are accounted for separately.

| Dense signal parameters | RTX 3090 microbatch | RTX 3090 examples/s | v5e-1 microbatch | v5e-1 examples/s | Warm speed ratio |
|---|---:|---:|---:|---:|---:|
| 3,614,473 | 256 | 4,128 | 128 | 23,624 | 5.72× |
| 117,107,081 | 64 | 218 | 32 | 1,617 | 7.42× |

The selected pilot compiled in 7.61 seconds on the TPU and 25.22 seconds on the 3090. The selected 117M batch compiled in 17.75 and 39.15 seconds respectively. Across the pack, 99.95% of fixed positions were valid. The dense pilot Transformer validated at about 26,000 examples/s with correct, mismatched and erased sources evaluated together. At 26M the dense signal model validated at about 10,833 examples/s on development and 9,552 on confirmation. The installed JAX versions differed across devices. The measured runs normalized masked auxiliary terms within a microbatch. Subsequent code uses effective-batch denominators and has a gradient-equivalence regression test. Within each reported architecture pair the microbatch also matched. [All batch sizes, compilation times and repeated timings](throughput.json)

Four allocation attempts for v6e-1 ended in a timeout or provider gateway/service errors. Google provisioned the authorized v5e fallback. The CLI confirmed its release and the account balance moved from 198.77 to 196.20 compute units. V6e speed remains unmeasured.

## Which modality actually carried information?

With the medium signal model on 749 held-out GRID clips from eight people, text-to-audio normalized MSE was **0.2479** with the supplied text, **0.2573** with another prompt and **0.3702** with erased text. For audio-to-video, correct and wrong audio both gave approximately **0.0700**. For video-to-audio, the same model gave **0.2650** with correct video, **0.3103** with wrong video and **0.2517** with erased video. The model detects some mismatches while a domain-average prediction can still win. [Frozen conditional errors, temporal differences and speaker-group intervals](confirmation.json)

Eight additional medium runs compared cross-only supervision with a centered motion loss and ±8-tick foils from the same training recording. The table reports normalized audio-feature MSE on the eight confirmation speakers. Each row uses 256 updates, effective batch 256 and the same fixed 0.0003 learning rate.

| Core | Seed | Cross-only, correct video | Timing loss, correct video | Timing loss, wrong video | Timing loss, erased video |
|---|---:|---:|---:|---:|---:|
| Dense GPT-2-style | 81 | 0.2839 | 0.2639 | 0.3420 | 0.2663 |
| Dense GPT-2-style | 82 | 0.2791 | 0.2584 | 0.3330 | 0.2645 |
| Dense signal | 81 | 0.2719 | 0.2961 | 0.3633 | 0.2669 |
| Dense signal | 82 | 0.2812 | 0.3052 | 0.3689 | 0.2656 |

For the dense Transformer, correct video reduced error by about **7%** relative to its cross-only recipe. The speaker-bootstrap intervals for its gain over wrong video were positive in both seeds. Both intervals for its smaller gain over erased video included zero. For the signal core, timing discrimination strengthened while correct-input prediction worsened. Across the four timing-loss runs, swapping the audio changed mean video error by less than 0.0001. [Held-out intervention and intervals](motion-confirmation-summary.json)

An external frozen Whisper tiny.en evaluator scored four deterministic clips per held-out person. Original audio had **49.5%** mean clip WER and the measured audio codec reconstruction had **58.3%**. All eight generated conditions, spanning the pilot dense models and the medium GPT-2-style control and timing model, scored **100%** from text or video. This is a calibrated 32-clip diagnostic on restricted grammar. It supplies no evidence of intelligible open-ended speech. [Speech-content audit](speech-content.json)

## Scaling, transfer and action history

At the same 65,536 presentations, the dense Transformer's two-seed mean development video-to-audio error fell from **0.361 to 0.315 to 0.294** at 3.61M, 26.04M and 117.11M parameters. The signal core moved from **0.342 to 0.312 to 0.307**. At the middle size a quarter-data run saw 23,794 unique records, compared with 54,217 visited in the full pool. Neither architecture showed a consistent full-pool improvement across seeds at this short budget. This is an early learning-curve measurement. It cannot establish a data scaling law or convergence at 117M. [Equal-exposure development measurements](development.json)

Frozen eight-shot readouts received the same support examples in each comparison. The dense signal pilots scored **19.8% and 19.8%** balanced accuracy on unseen ESC-50 audio, with a **19.75%** raw-feature control. On CIFAR-10 they scored **18.9% and 18.4%**, with a **16.2%** raw control. The 2.04M grouped signal scored **20.1% and 20.0%** on CIFAR-10 versus **19.0% and 18.2%** for the equally sized dense Transformer. Its ESC-50 comparison changed direction across seeds. The 26M signal model at 1,024 updates scored 24.5% on ESC-50, while the 117M signal model at 512 updates scored 22.25%; their budgets differ. [All 1/4/8-shot outputs](confirmation.json)

On the pilot confirmation speakers, applying an eight-shot command head learned on audio directly to visual states was close to the 25% four-command chance rate. The follow-up tied the 128-column alignment basis to every decoder at every tick. In **278 clips from three additional G4-unused speakers**, mean bidirectional readout transfer for the dense Transformer moved from **23.95% to 25.83%** and **27.15% to 29.41%**. For the signal core the changes were **26.11% to 25.30%** and **25.09% to 25.58%**. This localizes one limitation of the independent projection in the Transformer. The combined bottleneck-and-loss intervention and small speaker count leave the underlying mechanism open. [Independent speaker results](new-speaker-summary.json)

The causal numeric encoder sees the current robot state and commands only through the preceding tick. A corrected DROID audit gives every predictor this same information cutoff. Across the two dense signal pilots the normalized next-command MSE was **0.108–0.111**. Repeating the last available command scored **0.075**, and constant-velocity extrapolation scored **0.048**. The original next-persistence field had one extra command available and is labelled privileged in the audit. The recorded actions do not establish closed-loop decision quality. [Action history controls](confirmation.json)

## Continuous outputs and research decision

The dense signal and dense Transformer each processed one continuous held-out AMI recording for **1,600 ticks, 68.27 seconds**, carrying model state between blocks. The audio stream generated 16×16 visual frames, the visual stream generated 100-band audio features, and a separate original text stream generated both. A frozen Vocos codec rendered the waveform. The clips contain generated target channels only. Each replay block buffered 64 ticks, about 2.73 seconds, in addition to acoustic analysis and rendering latency. For the signal video-to-audio stream, carrying state reduced MSE from **0.569** with repeated resets to **0.441**, while erasing video scored **0.309**. Memory affected replay, but useful continuous speech generation remains unproven. [Live controls, file hashes and durations](confirmation.json)

Pursue timing supervision first with the dense Transformer. Test speech-unit or phonetic targets and explicit source reliability against the erased-input baseline, with word recovery as the promotion criterion. For vision, the next useful target is speaker-independent motion with a measured content or landmark readout. For decisions, use an action-conditioned predictor with an explicit residual over causal persistence, then test interventions and returns in a simulator. Increasing width alone has not resolved these failures.

All **40 inference endpoints** were independently verified in eight Drive archives. Full optimizer states were verified for **25**. The other 15 support inference and fresh fine-tuning. Exact optimizer continuation for them remains unverified. Future stage admission reserves both wall time and bytes for outstanding full-state transfers. [Per-endpoint artifact audit](final-artifact-audit.json)

## Reproduce a saved output

On a compute machine with the private `slwm-drive` remote, restore the pilot inference group into a checkpoint directory. The restored FP32 model archive includes its feature normalization and the frozen optimizer endpoint identity.

```sh
python scripts/train/g4_fetch_weights.py --root /mnt/slwm-checkpoints-a/SLWM/g4 --group g4-v5e-pilot
```

The Fedora stage service checks training progress every 30 seconds and exposes its history in the `watch` window of `tmux attach -t slwm-g4`. Its small current-state file is `~/.local/state/slwm/research-stages.json`. The check does not allocate cloud hardware.
