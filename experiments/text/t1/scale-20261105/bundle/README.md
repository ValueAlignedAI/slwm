# Early GPT-2-scale text checkpoints

`gpt2_seed81.pt` and `slwm_seed81.pt` contain selected inference weights/configs.
All four full optimizer checkpoints remain in the run directories. Both models
start from random weights and see 16.8M tokens from a 2.6B-token prepared corpus.
These are early learning artifacts, not fluent or factual assistants.

```bash
python -m exploration.scale_text_infer --checkpoint BUNDLE/slwm_seed81.pt \
  --prompt "The purpose of a model is" --tokens 64 --device cpu
```

Replace `BUNDLE` with this directory. No training corpus is needed for inference.
Read [the report](../report.md), [compute diagnostic](../compute_diagnostic.md)
and all fixed unselected continuations before drawing conclusions. Only 512-token
contexts were trained in this stage; a larger positional table is not evidence
of 1024-token quality. The trained speech models are separate weights.
