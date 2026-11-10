# Compute recovery on November 10

Fedora returned after a reboot. The three registered filesystem UUIDs and the two legacy checkpoint bind mounts were verified. The 3090 restored its first 116.50M anchor from optimizer update 3,072. Google subsequently assigned two native v5e-1 chips after v6e remained unavailable. They train independent same-chip hybrid/dense pairs at 116.50M and 202.35M. At 09:24 UTC Google reported two assignments, 190.85 compute units remaining and a combined rate of 6.18 units/hour. [Verified TPU comparisons](../../t6/tpu-scale-20261110/matched.md) · [Frozen speaker transfer](../../t6/tpu-scale-20261110/transfer-and-decisions.md)

## Earlier state at 07:50 UTC

Twelve G5 runs finished on the RTX 3090. They comprise four small pilots at 192 updates, four small longer-schedule models at 2,048, and four width-512 residual models at 4,096. Both inference weights and optimizer state for all four medium models have independent verified Fedora HDD copies. Eight small checkpoint sets had verified Drive receipts. Google Drive subsequently returned a per-project query-rate limit to the file-scoped client.

The 116,502,025-parameter anchor launched on the 3090. Its first hybrid seed reached a directly observed 3,020 of 8,192 updates at approximately 904 examples per second. Fedora became unreachable from the Mac at approximately 07:32 UTC. The local recovery worker is configured to check the actual SSD and HDD mounts and resume the registered recipe from the last completed checkpoint once SSH is available. The four-arm anchor and terminal tests require an estimated 2.5–3 additional hours of available 3090 time from the last observed position.

The official Colab CLI repeatedly returned HTTP 502 on v6e-1 allocation. At 07:50 UTC Google reported zero active assignments, a 196.19 compute-unit balance, and a zero hourly rate. No named G5 TPU runtime had been observed. The starting account balance was 196.20. Two capped lanes will retry at approximately 08:47 UTC. They can train the matched anchor in parallel when v6e hardware and the verified input packs are accessible. The aggregate ceiling is 100 units. The previous recurring chat notification was removed.

[Matched results](report.md) · [Anchor gate and weight hashes](metrics/medium-residual-decision.json) · [Sealed test procedure](residual.md)
