# Interim text inference

Four independent GPT-2/SLWM language checkpoints selected at433.75M training tokens. These are not the shared multimodal weights. Registered500M final evaluation is pending. Original optimizer/RNG resumes remain on Fedora.

Each run contains `model.pt`, a SHA256 manifest and one fixed64-token continuation. All four raw outputs are in `samples.json`, including poor ones. Actual CPU loading and sampling passed from the included17-file source snapshot without loading any corpus.

From `source_snapshot/`:

```bash
python -m exploration.scale_text_infer --checkpoint ../EXP-T1-232/model.pt --prompt "A robot observes a cup on the table. Before moving, it should" --tokens 64 --seed 81 --device cpu
```

No factual, reasoning or planning claim follows from these early examples.
