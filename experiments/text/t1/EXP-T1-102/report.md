# Sprint T1 PyTorch Report — EXP-T1-102

- Model variant: `slwm_text_only`
- Tokenizer: `gpt2_bpe`
- Prepared corpus: `artifacts/t1_text_code/gpt2_bpe_larger_local_v0`
- Validation loss: `9.221770763397217`
- Validation perplexity: `10114.95967962054`
- Validation tokens evaluated: `4096` of `15559` available
- Throughput tokens/s: `5341.088488775351`
- Max RSS memory MB: `1358.5625`
- MPS allocated memory MB: `1599.7763671875`
- Checkpoint: `experiments/text/t1/EXP-T1-102/checkpoint.pt`

## Scope
Text/code only. No audio or visual data was loaded or trained in this sprint run.

## Claim limits
GPT-2-BPE prepared-corpus T1 run: report validation loss/PPL, samples, throughput, memory, and exact settings only. Claim scope is 'gpt2_size_limited_steps_larger_local_benchmark_not_converged_full_gpt2_training'; do not claim converged GPT-2-scale quality unless the registered train-token budget supports it.
