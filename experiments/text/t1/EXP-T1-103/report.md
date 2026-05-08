# Sprint T1 PyTorch Report — EXP-T1-103

- Model variant: `slwm_text_only_no_spectral`
- Tokenizer: `gpt2_bpe`
- Prepared corpus: `artifacts/t1_text_code/gpt2_bpe_larger_local_v0`
- Validation loss: `9.140068531036377`
- Validation perplexity: `9321.40391541339`
- Validation tokens evaluated: `4096` of `15559` available
- Throughput tokens/s: `6053.176885814016`
- Max RSS memory MB: `1350.25`
- MPS allocated memory MB: `1591.446533203125`
- Checkpoint: `experiments/text/t1/EXP-T1-103/checkpoint.pt`

## Scope
Text/code only. No audio or visual data was loaded or trained in this sprint run.

## Claim limits
GPT-2-BPE prepared-corpus T1 run: report validation loss/PPL, samples, throughput, memory, and exact settings only. Claim scope is 'gpt2_size_limited_steps_larger_local_benchmark_not_converged_full_gpt2_training'; do not claim converged GPT-2-scale quality unless the registered train-token budget supports it.
