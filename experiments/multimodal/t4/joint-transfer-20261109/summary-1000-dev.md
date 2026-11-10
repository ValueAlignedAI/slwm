# G3 joint transfer

Fixed 1,000-update dev comparison. Each row uses the same 32 labels per class within task.

| Profile | Seed | Spoken digits | Fashion images | Banking intents | Text NLL | Active training min |
|---|---:|---:|---:|---:|---:|---:|
| gpt2-dense | 81 | 17.8% | 73.7% | 68.5% | 4.298 | 3.8 |
| signal-dense | 81 | 25.6% | 74.6% | 70.5% | 4.164 | 5.0 |
| signal-sparse | 81 | 30.8% | 74.5% | 70.3% | 4.155 | 8.0 |
| gpt2-wide | 81 | 21.8% | 73.1% | 69.0% | 4.383 | 3.9 |
| signal-wide | 81 | 26.4% | 74.0% | 70.8% | 4.252 | 5.0 |
| gpt2-dense | 82 | 43.6% | 72.5% | 69.1% | 4.289 | 3.4 |
| signal-dense | 82 | 45.4% | 74.2% | 70.5% | 4.169 | 4.5 |
| signal-sparse | 82 | 33.2% | 73.5% | 70.9% | 4.162 | 7.4 |
| gpt2-wide | 82 | 37.6% | 73.4% | 68.7% | 4.353 | 3.5 |
| signal-wide | 82 | 42.8% | 74.5% | 71.2% | 4.234 | 4.7 |

| Frozen control | Seed | Spoken digits | Fashion images | Banking intents |
|---|---:|---:|---:|---:|
| raw | 0 | 28.8% | 75.3% | 67.7% |
| gpt2-dense | 81 | 31.2% | 67.6% | 68.3% |
| signal-dense | 81 | 23.6% | 68.9% | 69.8% |
| signal-sparse | 81 | 23.6% | 68.9% | 69.8% |
| gpt2-wide | 81 | 31.2% | 67.6% | 68.3% |
| signal-wide | 81 | 23.6% | 68.9% | 69.8% |
| gpt2-dense | 82 | 37.8% | 70.1% | 68.7% |
| signal-dense | 82 | 37.8% | 69.6% | 70.4% |
| signal-sparse | 82 | 37.8% | 69.6% | 70.4% |
| gpt2-wide | 82 | 37.8% | 70.1% | 68.7% |
| signal-wide | 82 | 37.8% | 69.6% | 70.4% |
| G2-201 | 0 | 19.2% | 58.7% | 5.7% |
| G2-202 | 0 | 19.4% | 60.7% | 5.8% |
| G2-203 | 0 | 15.2% | 63.6% | 6.6% |
| G2-204 | 0 | 15.8% | 61.0% | 5.5% |

Probes fit only labelled support. The few-shot curves, raw and unadapted controls, paired uncertainty, G2 anchor and retrieval are available in the metric files. Development outcomes remain separate from endpoint confirmation. FSDD has two final speakers.
