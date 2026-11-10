# Matched hardware throughput

Both devices executed the same joint model, real data order, optimizer, learning-rate schedule and effective batch of 256. Each speed is the median of two warm four-update blocks after synchronizing the full optimizer state. The 3090 sweep tested microbatches 32, 64 and 128. The v5e sweep tested 32 and 64. Compilation was outside the warm measurements. The installed JAX versions differed. [Full candidate measurements](hardware-throughput.json)

| Parameters | Core | 3090, examples/s | v5e-1, examples/s | Ratio, best tested | Ratio at microbatch 64 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 116.50M | hybrid | 1,156 | 2,783 | 2.41× | 2.52× |
| 116.50M | gpt2 | 1,172 | 2,683 | 2.29× | 2.40× |
| 202.35M | hybrid | 685 | 1,706 | 2.49× | 2.59× |
| 202.35M | gpt2 | 698 | 1,615 | 2.31× | 2.66× |

Both v5e lanes used the fastest tested 64-example microbatch. The RTX 3090 favored 128. These are measured single-chip training speeds. Cold file reads, compilation, evaluation, and durable artifact transfer increase total elapsed time.
