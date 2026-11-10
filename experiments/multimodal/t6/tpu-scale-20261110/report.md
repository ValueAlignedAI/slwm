# G6 TPU scale measurements

[Matched learning curves, artifact identities and size controls](matched.md) · [Frozen cross-modal transfer and terminal decision readouts](transfer-and-decisions.md)

The first 202,346,377-parameter models trained on Google Colab's single-chip v5e. Both the hybrid and all-dense GPT-2-style control completed 128 optimizer updates at batch 256 on the same frozen nine-family dataset. That is 32,768 examples and 2,097,152 multimodal positions per arm. Their exact optimizer states and inference files were independently verified on both Fedora checkpoint HDDs. Each optimizer archive is 2,428,259,696 bytes. [Registered design](protocol.md) · [Model hashes, full controls and paired intervals](metrics/v5e-511-larger-128.json)

| Development result at 128 updates, seed 511 | Hybrid | Dense GPT-2 | Physical control |
| --- | ---: | ---: | ---: |
| Endpoint error across 256 new simulated scenes, pixels ↓ | 21.60 | 21.50 | Fitted action prior 21.42 |
| Visual contact average precision across three recording dates ↑ | 0.05819 | 0.05806 | Raw frame motion 0.05814 |

At this early exposure the joint win criterion is unmet. The two model readouts begin as the same trained physical prior, so the learned residual must improve that prior for a representation claim. The independently archived artifacts live under `/mnt/slwm-checkpoints-a/SLWM/g6/<run>/step-000128/` and the same relative address on checkpoint disk B. Each directory contains `inference.npz`, `resume.npz`, and a verification receipt.

In the first warm training window, 64-example microbatches at effective batch 256 delivered roughly 1,707 examples/s for the hybrid and 1,616 for dense. Cold host batch reads later reached 2–4 seconds for four updates, versus 13 ms when the same inputs were cached. All 8,462,712,314 bytes of train/development arrays were subsequently copied to shared memory and checked against their immutable per-array hashes. Full-stage elapsed time includes compilation, cold inputs and checkpointing overhead in addition to warm model updates.

The second native v5e now supplies an independent training seed. At 09:24 UTC the two assignments together billed 6.18 Colab units/hour. The first restoration drill reconstructed all four real 2,048-update optimizers for seed 511, including both parameter sizes and both architectures, and checked their registered identities and counters. The Mac carried no model payload on disk. [Restoration evidence](restore-drill.json)

The 202.35M hybrid at 8,192 updates generated all three 43.69-second continuous outputs for seed 511. Against the actual held recording, the generated sound’s log-energy correlation was 0.0792, below the 0.0905 within-recording shifted 95th percentile. Generated frame motion correlated 0.1263 with observed motion, above the corresponding 0.1094 threshold on this one clip. The grayscale 16×16 frames retain little recognizable scene detail. This is narrow timing evidence. The final speech decoder and fresh goal-reaching controls are separate tests. The full model and source checksums are in the media manifest. [Continuous media evidence](media-hybrid-511.json)

On the identical continuous source and exposure, the dense model’s generated sound envelope correlated −0.0304 with the recording; its shifted 95th percentile was 0.0778. The generated motion correlation was 0.0227 versus a shifted 95th percentile of 0.0755. The hybrid’s modest motion effect on this one clip did not extend to both output modalities or to the dense control. [Dense continuous media evidence](media-dense-511.json)

## Independent data staged for the next speech comparison

The full LibriSpeech train-clean-360 release passed the publisher’s checksum. It contains 104,014 paired utterances across 921 speakers. Removing 21,678 existing training utterance IDs and 7,197 additional utterances from previously held speaker groups leaves **75,139 new audio–transcript pairs from 665 speakers absent from earlier training**. Fedora holds a compressed manifest referencing the verified original media. The high-scale G6 comparisons used their original locked data throughout. The recovery procedure reacquired one 8 KiB hole in the pre-reboot prefix before accepting the complete 23.05 GB archive. [Dataset accounting](libri360-card.json) · [Integrity evidence](libri360-integrity.json) · [Original release](https://openslr.org/12/)
