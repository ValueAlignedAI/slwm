# G6 TPU scale measurements

The first 202,346,377-parameter models trained on Google Colab's single-chip v5e. Both the hybrid and all-dense GPT-2-style control completed 128 optimizer updates at batch 256 on the same frozen nine-family dataset. That is 32,768 examples and 2,097,152 multimodal positions per arm. Their exact optimizer states and inference files were independently verified on both Fedora checkpoint HDDs. Each optimizer archive is 2,428,259,696 bytes. [Registered design](protocol.md) · [Model hashes, full controls and paired intervals](metrics/v5e-511-larger-128.json)

| Development result at 128 updates, seed 511 | Hybrid | Dense GPT-2 | Physical control |
| --- | ---: | ---: | ---: |
| Endpoint error across 256 new simulated scenes, pixels ↓ | 21.60 | 21.50 | Fitted action prior 21.42 |
| Visual contact average precision across three recording dates ↑ | 0.05819 | 0.05806 | Raw frame motion 0.05814 |

At this early exposure the joint win criterion is unmet. The two model readouts begin as the same trained physical prior, so the learned residual must improve that prior for a representation claim. The independently archived artifacts live under `/mnt/slwm-checkpoints-a/SLWM/g6/<run>/step-000128/` and the same relative address on checkpoint disk B. Each directory contains `inference.npz`, `resume.npz`, and a verification receipt.

In the first warm training window, 64-example microbatches at effective batch 256 delivered roughly 1,707 examples/s for the hybrid and 1,616 for dense. Cold host batch reads later reached 2–4 seconds for four updates, versus 13 ms when the same inputs were cached. All 8,462,712,314 bytes of train/development arrays were subsequently copied to shared memory and checked against their immutable per-array hashes. Full-stage elapsed time includes compilation, cold inputs and checkpointing overhead in addition to warm model updates.
