# G3 joint transfer

Fixed 100-update dev comparison. Each row uses the same 32 labels per class within task.

| Profile | Seed | Spoken digits | Fashion images | Banking intents | Text NLL | Active training min |
|---|---:|---:|---:|---:|---:|---:|
| gpt2-dense | 81 | 25.2% | 68.8% | 67.1% | 4.298 | 0.5 |
| signal-dense | 81 | 22.8% | 70.2% | 70.1% | 4.159 | 0.5 |
| signal-sparse | 81 | 29.4% | 70.0% | 69.8% | 4.156 | 0.8 |
| gpt2-wide | 81 | 23.4% | 69.9% | 66.8% | 4.339 | 0.3 |
| signal-wide | 81 | 25.8% | 70.1% | 70.1% | 4.184 | 0.3 |
| gpt2-dense | 82 | 36.8% | 69.9% | 68.2% | 4.279 | 0.3 |
| signal-dense | 82 | 30.6% | 70.4% | 70.3% | 4.171 | 0.3 |
| signal-sparse | 82 | 29.6% | 70.7% | 70.6% | 4.157 | 0.3 |
| gpt2-wide | 82 | 36.0% | 69.8% | 68.7% | 4.337 | 0.2 |
| signal-wide | 82 | 31.0% | 70.3% | 70.4% | 4.203 | 0.3 |

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
