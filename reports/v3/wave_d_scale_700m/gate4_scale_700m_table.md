### Validation / Dev Suite

| Arm | Seeds | Core Params | State (MiB) | WT-103 Val BPB ↓ | ESC-50 Top-1 ↑ | Clotho A→T ↑ | MSR-VTT V→T ↑ | VGGSound V→A ↑ | Robotics Vid ↓ | Robotics Act ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `single_bank_gated_deltanet_wy` | 1 | 703,439,960 | 4.502 | 2.2322 | 0.7250 | 0.0090 | 0.0127 | 0.0227 | 0.1021 | 7.8268 |
| `slwm_v3_routed_c32` | 1 | 703,149,536 | 7.502 | 2.2528 | 0.7050 | 0.0139 | -0.0006 | 0.0383 | 0.1014 | 8.1566 |

### Sealed-Test Official External SoTA Suite

| Arm | Seeds | WT-103 Test BPB ↓ | LAMBADA Acc ↑ | HumanEval BPB ↓ / pass@1 ↑ | MBPP BPB ↓ / pass@1 ↑ | ESC-50 Fold-5 Top-1 ↑ | LibriSpeech WER ↓ | Clotho A→T ↑ | MSR-VTT V→T ↑ | VGGSound V→A ↑ | Robotics Vid ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `single_bank_gated_deltanet_wy` | 1 | 2.2709 | 0.0035 | 1.444 / 0.000 | 1.675 / 0.000 | 0.4150 | 1.5493 | 0.0045 | -0.0077 | -0.0044 | 0.1067 |
| `slwm_v3_routed_c32` | 1 | 2.2967 | 0.0064 | 1.450 / 0.000 | 1.675 / 0.000 | 0.4175 | 1.5559 | 0.0062 | 0.0163 | 0.0092 | 0.1051 |
