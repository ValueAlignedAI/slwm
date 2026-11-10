# G3 joint transfer

Fixed 8,000-update test comparison. Each row uses the same 32 labels per class within task.

| Profile | Seed | Spoken digits | Fashion images | Banking intents | Text NLL | Active training min |
|---|---:|---:|---:|---:|---:|---:|
| gpt2-dense | 81 | 54.0% | 75.7% | 67.3% | 4.284 | 28.9 |
| signal-dense | 81 | 61.2% | 76.2% | 69.4% | 4.158 | 38.6 |
| signal-sparse | 81 | 59.6% | 75.1% | 70.0% | 4.158 | 66.8 |
| gpt2-wide | 81 | 47.9% | 75.8% | 66.6% | 4.294 | 33.2 |
| signal-wide | 81 | 55.4% | 75.9% | 69.6% | 4.161 | 42.0 |
| gpt2-dense | 82 | 47.3% | 75.8% | 68.9% | 4.271 | 29.0 |
| signal-dense | 82 | 55.4% | 75.5% | 69.8% | 4.165 | 37.2 |
| signal-sparse | 82 | 56.9% | 75.0% | 68.9% | 4.161 | 62.6 |
| gpt2-wide | 82 | 46.3% | 75.0% | 67.8% | 4.276 | 29.1 |
| signal-wide | 82 | 64.1% | 75.9% | 68.8% | 4.169 | 38.9 |

| Frozen control | Seed | Spoken digits | Fashion images | Banking intents |
|---|---:|---:|---:|---:|
| raw | 0 | 66.4% | 74.5% | 69.3% |
| gpt2-dense | 81 | 45.1% | 68.1% | 67.7% |
| signal-dense | 81 | 44.1% | 69.6% | 70.0% |
| signal-sparse | 81 | 44.1% | 69.6% | 70.0% |
| gpt2-wide | 81 | 45.1% | 68.1% | 67.7% |
| signal-wide | 81 | 44.1% | 69.6% | 70.0% |
| gpt2-dense | 82 | 36.1% | 68.8% | 68.9% |
| signal-dense | 82 | 49.5% | 69.5% | 70.5% |
| signal-sparse | 82 | 49.5% | 69.5% | 70.5% |
| gpt2-wide | 82 | 36.1% | 68.8% | 68.9% |
| signal-wide | 82 | 49.5% | 69.5% | 70.5% |
| G2-201 | 0 | 22.5% | 59.1% | 5.7% |
| G2-202 | 0 | 24.2% | 61.5% | 5.6% |
| G2-203 | 0 | 21.2% | 64.7% | 6.1% |
| G2-204 | 0 | 19.6% | 61.9% | 5.4% |

Probes fit only labelled support. The few-shot curves, raw and unadapted controls, paired uncertainty, G2 anchor and retrieval are available in the metric files. Development outcomes remain separate from endpoint confirmation. FSDD has two final speakers.
