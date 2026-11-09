# Using G3 inference artifacts

A checkpoint consists of the original text parent, one joint delta and its adjacent card. For 12-layer seed 81, dense GPT-2 uses EXP-T1-231 and the two signal variants use EXP-T1-232. Seed 82 uses EXP-T1-233 and EXP-T1-234. The 6-layer study has separately derived, hash-pinned parents. The fixed core, jointly updated adapters and typed edges together form the inference model. Both parent and delta checksums are verified at load time.

From the G3 checkout, with the existing research Python dependencies installed, pass the paths on your machine. For example:

```bash
python -m exploration.joint_infer \
  --parent ../slwm/experiments/text/t1/500m-20261106/EXP-T1-232/best.pt \
  --delta experiments/multimodal/t4/joint-transfer-20261109/signal-sparse-81/step-100.pt \
  --prompt 'A sound behind the door' --tokens 24 --output /tmp/g3-text.json
```

For signal inference replace the prompt and token options with `--packets observed.npz --output /tmp/g3-future.npz`. Accepted keys are `image[B,T,192]`, `audio[B,T,100]` and `numeric[B,T,64]` with Boolean `present[B,T,64]`. Pass float32 arrays on the joint training scales. Static RGB uses sixteen flattened 8-by-8 color patches from a 32-by-32 image. Temporal video uses one 8-by-8 color thumbnail per tick. Audio uses normalized log mel bands. Numeric uses the registered physical channel layout and training-role normalizers. The packet file contains only received observations. The output contains the causal shared field and each supplied modality's next-position prediction. The CLI performs greedy decoding or direct signal prediction without weight updates.

The `verified-*.json` files tie copied milestone deltas to the exact parents and exposure digests. GitHub contains code, hashes and aggregate results. Large weights are held on the research machines.
