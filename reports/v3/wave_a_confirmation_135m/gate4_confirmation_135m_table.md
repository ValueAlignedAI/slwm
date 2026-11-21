### Validation / Dev Suite

| Arm | Seeds | Core Params | State (MiB) | WT-103 Val BPB ↓ | ESC-50 Top-1 ↑ | Clotho A→T ↑ | MSR-VTT V→T ↑ | VGGSound V→A ↑ | Robotics Vid ↓ | Robotics Act ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `single_bank_gated_deltanet_wy` | 1 | 137,554,512 | 2.001 | 2.3386 | 0.7000 | 0.0066 | 0.0042 | 0.0437 | 0.1147 | 8.1724 |
| `sliding_window_gqa_eqbyte` | 1 | 137,845,184 | 2.002 | 2.3133 | 0.7050 | 0.0053 | 0.0035 | 0.0252 | 0.1274 | 8.2838 |
| `slwm_v3_routed_c16` | 1 | 137,846,080 | 3.001 | 2.3237 | 0.7250 | 0.0067 | 0.0069 | 0.0318 | 0.1139 | 8.3278 |

### Sealed-Test Official External SoTA Suite

| Arm | Seeds | WT-103 Test BPB ↓ | LAMBADA Acc ↑ | HumanEval BPB ↓ / pass@1 ↑ | MBPP BPB ↓ / pass@1 ↑ | ESC-50 Fold-5 Top-1 ↑ | LibriSpeech WER ↓ | Clotho A→T ↑ | MSR-VTT V→T ↑ | VGGSound V→A ↑ | Robotics Vid ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `single_bank_gated_deltanet_wy` | 1 | 2.3760 | 0.0033 | 1.635 / 0.000 | 1.833 / 0.000 | 0.5250 | 1.5950 | 0.0071 | -0.0067 | -0.0338 | 0.1195 |
| `sliding_window_gqa_eqbyte` | 1 | 2.3382 | 0.0184 | 1.567 / 0.000 | 1.731 / 0.000 | 0.5300 | 1.5376 | 0.0024 | -0.0029 | -0.0286 | 0.1308 |
| `slwm_v3_routed_c16` | 1 | 2.3575 | 0.0012 | 1.616 / 0.000 | 1.867 / 0.000 | 0.5600 | 1.5772 | 0.0051 | -0.0431 | -0.0056 | 0.1179 |
