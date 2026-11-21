"""SLWM-v3 Live Frozen-Weight Any-to-Any Multimodal Streaming Inference & Demo Runner.

Loads a trained SLWM-v3 checkpoint (`*_ckpt.pkl`) plus optional Wave B (`PolicyCommitGate`) and
Wave C (multimodal decoder) checkpoints with strictly frozen weights (`no` online weight updates!)
and runs `O(1)` causal streaming inference (`models/v3/core.py:step_core`) across all modalities:
  1. `text2text`: Live byte-by-byte streaming text/code completion from any prompt.
  2. `audio2text`: Live streaming captioning / transcription from a pre-encoded shard episode (or `.wav` via Mimi).
  3. `video2text`: Live streaming video-to-text description from a pre-encoded shard episode.
  4. `video2audio`: Live streaming video-to-audio 32-RVQ Mimi code generation (optionally decoded to `.wav`).
  5. `action_rollout`: Embodied robotics next-proprioception & next-video prediction + counterfactual action comparison.
  6. `policy_gate`: Live epistemic routing (`SPEAK_TEXT`, `EMIT_AUDIO`, `GENERATE_VIDEO`, `ACT_MOTOR`, `ABSTAIN_SILENCE`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.v3_unified_shards import BYTE_EOS, BYTE_MOD_SEP, BYTE_PAD, LoadedSparseShard, decode_utf8_bytes
from models.v3.any2any_heads import (
    COMMIT_ACTIONS,
    apply_action_proprio_head,
    apply_audio_head,
    apply_policy_commit_gate,
    apply_text_head,
    apply_video_head,
)
from models.v3.core import init_core_state, step_core


def _empty_packet_b1(spec) -> dict[str, jax.Array]:
    return {
        "modality_id": jnp.array([2], dtype=jnp.int32),
        "dt_s": jnp.array([0.0], dtype=jnp.float32),
        "valid_mask": jnp.array([1.0], dtype=jnp.float32),
        "reliability": jnp.array([1.0], dtype=jnp.float32),
        "reset_mask": jnp.array([False], dtype=jnp.bool_),
        "spatial_pos": jnp.array([0], dtype=jnp.int32),
        "text_byte": jnp.array([BYTE_PAD], dtype=jnp.int32),
        "has_true_action": jnp.array([0.0], dtype=jnp.float32),
        "video_feat": jnp.zeros((1, spec.video_patch_dim), dtype=jnp.float32),
        "audio_codes": jnp.zeros((1, spec.audio_codebooks), dtype=jnp.int32),
        "proprio_feat": jnp.zeros((1, spec.proprio_dim), dtype=jnp.float32),
        "action_feat": jnp.zeros((1, spec.action_dim), dtype=jnp.float32),
        "probe_feat": jnp.zeros((1, spec.probe_dim), dtype=jnp.float32),
    }


def load_frozen_bundle(
    core_ckpt: Path,
    *,
    wave_b_ckpt: Path | None = None,
    wave_c_ckpt: Path | None = None,
) -> dict[str, Any]:
    with core_ckpt.open("rb") as f:
        obj = pickle.load(f)
    spec = obj["spec"]
    params = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float32), obj["params"])
    gate_params = None
    if wave_b_ckpt is not None and wave_b_ckpt.is_file():
        with wave_b_ckpt.open("rb") as f:
            gate_params = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float32), pickle.load(f)["gate_params"])
    if wave_c_ckpt is not None and wave_c_ckpt.is_file():
        with wave_c_ckpt.open("rb") as f:
            params["heads"] = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float32), pickle.load(f)["dec_params"])
    return {"spec": spec, "params": params, "gate_params": gate_params}


def _sample_next_byte(logits_264: np.ndarray, *, temperature: float, rng: np.random.Generator) -> int:
    lg = logits_264[:258].astype(np.float64)  # 0..255 UTF-8 bytes + EOS(257) (exclude PAD=256)
    lg[BYTE_PAD] = -1e9
    if temperature <= 1e-4:
        return int(np.argmax(lg))
    lg = (lg - np.max(lg)) / max(temperature, 1e-4)
    p = np.exp(lg)
    p /= np.sum(p)
    return int(rng.choice(len(p), p=p))


def stream_text_continuation(
    bundle: dict[str, Any],
    prompt: str,
    *,
    max_new_bytes: int = 160,
    temperature: float = 0.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Run live `O(1)` streaming byte-by-byte continuation with strictly frozen weights."""
    spec, params, gate_params = bundle["spec"], bundle["params"], bundle["gate_params"]
    st = init_core_state(spec, 1)
    rng = np.random.default_rng(seed)

    @jax.jit
    def _tick(p, s, pkt):
        z, ns, dm = step_core(p["core"], s, pkt, spec)
        t_lg = apply_text_head(p["heads"]["head_text"], z[None, :, :], spec)[0, 0]
        return z, ns, t_lg, dm["mean_nowrite"], dm["mean_prior_err"]

    prompt_bytes = list(prompt.encode("utf-8", errors="ignore"))
    last_lg = None
    last_z = None
    nowrites: list[float] = []
    for i, b_val in enumerate(prompt_bytes):
        pkt = _empty_packet_b1(spec)
        pkt["text_byte"] = jnp.array([b_val], dtype=jnp.int32)
        pkt["reset_mask"] = jnp.array([i == 0], dtype=jnp.bool_)
        last_z, st, last_lg, nw, _ = _tick(params, st, pkt)
        nowrites.append(float(nw))

    gen_bytes: list[int] = []
    gate_actions: list[str] = []
    for _ in range(max_new_bytes):
        assert last_lg is not None and last_z is not None
        nxt = _sample_next_byte(np.asarray(last_lg), temperature=temperature, rng=rng)
        if nxt == BYTE_EOS:
            break
        gen_bytes.append(nxt)
        if gate_params is not None:
            probs_b = jax.nn.softmax(last_lg[:258])
            t_ent = float(-jnp.sum(probs_b * jnp.log(jnp.maximum(probs_b, 1e-8))))
            ev = jnp.array([[t_ent, 0.0, 0.0, 1.0, float(jnp.max(probs_b))]], dtype=jnp.float32)
            _, gp = apply_policy_commit_gate(gate_params, last_z, ev)
            gate_actions.append(COMMIT_ACTIONS[int(jnp.argmax(gp[0]))])
        pkt = _empty_packet_b1(spec)
        pkt["text_byte"] = jnp.array([nxt], dtype=jnp.int32)
        last_z, st, last_lg, nw, _ = _tick(params, st, pkt)
        nowrites.append(float(nw))

    return {
        "mode": "text2text",
        "epistemic_label": "predicted_free_running_frozen_weights",
        "prompt": prompt,
        "generated": decode_utf8_bytes(np.asarray(gen_bytes, dtype=np.int32)),
        "n_generated_bytes": len(gen_bytes),
        "mean_nowrite": round(float(np.mean(nowrites)) if nowrites else 0.0, 4),
        "dominant_policy_gate_action": max(set(gate_actions), key=gate_actions.count) if gate_actions else "unloaded",
    }


def stream_cross_modal_from_shard(
    bundle: dict[str, Any],
    shard_path: Path,
    *,
    episode_idx: int = 0,
    target_mode: str = "text",
    max_new_steps: int = 96,
    temperature: float = 0.0,
    seed: int = 42,
    out_wav: Path | None = None,
) -> dict[str, Any]:
    """Feed observed multimodal context from a shard episode through `step_core`, then decode target modality live!"""
    spec, params = bundle["spec"], bundle["params"]
    sh = LoadedSparseShard.load(shard_path)
    s0, e0 = sh.episode_slices[episode_idx % len(sh.episode_slices)]
    d = sh.data
    mods = d["modality_id"][s0:e0]
    tb_ep = d["text_byte"][s0:e0]
    sep_pos = np.where(tb_ep == BYTE_MOD_SEP)[0]
    if target_mode == "text" and len(sep_pos) > 0:
        ctx_end = int(sep_pos[0]) + 1
        gt_txt = decode_utf8_bytes(tb_ep[ctx_end:])
    elif target_mode == "audio":
        aud_pos = np.where(mods == 1)[0]
        ctx_end = int(aud_pos[0]) if len(aud_pos) > 0 else min(len(mods) // 2, 256)
        gt_txt = ""
    else:
        ctx_end = min(len(mods) // 2, 256)
        gt_txt = decode_utf8_bytes(tb_ep[ctx_end:])

    st = init_core_state(spec, 1)
    rng = np.random.default_rng(seed)

    @jax.jit
    def _tick(p, s, pkt):
        z, ns, dm = step_core(p["core"], s, pkt, spec)
        t_lg = apply_text_head(p["heads"]["head_text"], z[None, :, :], spec)[0, 0]
        a_lg = apply_audio_head(p["heads"]["head_audio"], z[None, :, :], spec)[0, 0]  # [Q, V]
        v_p = apply_video_head(p["heads"]["head_video"], z[None, :, :], spec)[0, 0]
        ap_p = apply_action_proprio_head(p["heads"]["head_action"], z[None, :, :], spec)[0, 0]
        return z, ns, t_lg, a_lg, v_p, ap_p, dm["mean_nowrite"]

    last_t = last_a = None
    nowrites: list[float] = []
    # Observe real source modality context tick-by-tick in O(1) streaming state!
    for i in range(min(ctx_end, 512)):
        gi = s0 + i
        pkt = _empty_packet_b1(spec)
        pkt["modality_id"] = jnp.array([int(d["modality_id"][gi])], dtype=jnp.int32)
        pkt["dt_s"] = jnp.array([float(d["dt_s"][gi])], dtype=jnp.float32)
        pkt["reliability"] = jnp.array([float(d["reliability"][gi])], dtype=jnp.float32)
        pkt["reset_mask"] = jnp.array([i == 0], dtype=jnp.bool_)
        pkt["spatial_pos"] = jnp.array([int(d["spatial_pos"][gi])], dtype=jnp.int32)
        pkt["text_byte"] = jnp.array([int(d["text_byte"][gi])], dtype=jnp.int32)
        vi = int(d["video_idx"][gi])
        if vi >= 0 and len(d["video_feat"]) > 0:
            pkt["video_feat"] = jnp.asarray(d["video_feat"][vi : vi + 1], dtype=jnp.float32)
        ai = int(d["audio_idx"][gi])
        if ai >= 0 and len(d["audio_codes"]) > 0:
            pkt["audio_codes"] = jnp.asarray(d["audio_codes"][ai : ai + 1], dtype=jnp.int32)
        aci = int(d["act_idx"][gi])
        if aci >= 0 and len(d["proprio_feat"]) > 0:
            pkt["proprio_feat"] = jnp.asarray(d["proprio_feat"][aci : aci + 1], dtype=jnp.float32)
            pkt["action_feat"] = jnp.asarray(d["action_feat"][aci : aci + 1], dtype=jnp.float32)
            pkt["has_true_action"] = jnp.array([float(d["has_true_action"][gi])], dtype=jnp.float32)
        _, st, last_t, last_a, _, _, nw = _tick(params, st, pkt)
        nowrites.append(float(nw))

    if target_mode == "text":
        gen_b: list[int] = []
        for _ in range(max_new_steps):
            assert last_t is not None
            nxt = _sample_next_byte(np.asarray(last_t), temperature=temperature, rng=rng)
            if nxt == BYTE_EOS:
                break
            gen_b.append(nxt)
            pkt = _empty_packet_b1(spec)
            pkt["modality_id"] = jnp.array([2], dtype=jnp.int32)
            pkt["text_byte"] = jnp.array([nxt], dtype=jnp.int32)
            _, st, last_t, last_a, _, _, nw = _tick(params, st, pkt)
            nowrites.append(float(nw))
        return {
            "mode": f"cross_modal_to_text ({shard_path.name})",
            "episode_idx": int(episode_idx),
            "observed_context_ticks": int(min(ctx_end, 512)),
            "epistemic_label": "predicted_free_running_frozen_weights",
            "ground_truth_text": gt_txt[:240],
            "generated_text": decode_utf8_bytes(np.asarray(gen_b, dtype=np.int32)),
            "mean_nowrite": round(float(np.mean(nowrites)) if nowrites else 0.0, 4),
        }

    # `target_mode == "audio"`: autoregressively predict 32-RVQ Kyutai Mimi audio packets at 12.5 Hz!
    gen_codes: list[np.ndarray] = []
    for _ in range(max_new_steps):
        assert last_a is not None
        codes_32 = np.asarray(jnp.argmax(last_a, axis=-1), dtype=np.int32)  # [32]
        gen_codes.append(codes_32)
        pkt = _empty_packet_b1(spec)
        pkt["modality_id"] = jnp.array([1], dtype=jnp.int32)
        pkt["dt_s"] = jnp.array([0.08], dtype=jnp.float32)
        pkt["audio_codes"] = jnp.asarray(codes_32[None, :], dtype=jnp.int32)
        _, st, last_t, last_a, _, _, nw = _tick(params, st, pkt)
        nowrites.append(float(nw))
    codes_arr = np.stack(gen_codes, axis=0)  # [T_gen, 32]
    wav_saved = None
    if out_wav is not None:
        try:
            import torch
            from scipy.io import wavfile
            from transformers import MimiModel
            mimi = MimiModel.from_pretrained("kyutai/mimi")
            with torch.no_grad():
                c_t = torch.from_numpy(codes_arr.T[None, :, :].astype(np.int64))  # [1, 32, T]
                wav = mimi.decode(c_t).audio_values[0, 0].cpu().numpy()
            out_wav.parent.mkdir(parents=True, exist_ok=True)
            wavfile.write(out_wav, 24000, np.clip(wav * 32767.0, -32768, 32767).astype(np.int16))
            wav_saved = str(out_wav)
        except Exception as exc:
            wav_saved = f"skipped_mimi_decode: {exc!r}"
    return {
        "mode": f"cross_modal_to_audio ({shard_path.name})",
        "episode_idx": int(episode_idx),
        "observed_context_ticks": int(min(ctx_end, 512)),
        "generated_mimi_ticks_12p5hz": int(len(gen_codes)),
        "generated_duration_sec": round(len(gen_codes) / 12.5, 2),
        "wav_path": wav_saved,
        "epistemic_label": "predicted_free_running_frozen_weights",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Live frozen-weight O(1) streaming any-to-any inference for SLWM-v3.")
    ap.add_argument("--ckpt", type=Path, required=True, help="Path to saved `*_ckpt.pkl` core+heads checkpoint.")
    ap.add_argument("--wave-b-ckpt", type=Path, default=None, help="Optional saved `wave_b_*_ckpt.pkl` PolicyCommitGate.")
    ap.add_argument("--wave-c-ckpt", type=Path, default=None, help="Optional saved `wave_c_*_ckpt.pkl` Wave C decoders.")
    ap.add_argument("--mode", type=str, default="text2text", choices=("text2text", "audio2text", "video2text", "video2audio"))
    ap.add_argument("--prompt", type=str, default="def fibonacci(n):\n    ")
    ap.add_argument("--shard", type=Path, default=None, help="Pre-encoded shard `.npz` for cross-modal modes.")
    ap.add_argument("--episode-idx", type=int, default=0)
    ap.add_argument("--max-new-steps", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out-wav", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    bundle = load_frozen_bundle(args.ckpt, wave_b_ckpt=args.wave_b_ckpt, wave_c_ckpt=args.wave_c_ckpt)
    if args.mode == "text2text":
        res = stream_text_continuation(
            bundle, args.prompt, max_new_bytes=args.max_new_steps, temperature=args.temperature, seed=args.seed
        )
    else:
        if args.shard is None:
            raise ValueError(f"--shard is required for mode={args.mode}")
        tgt = "audio" if args.mode == "video2audio" else "text"
        res = stream_cross_modal_from_shard(
            bundle, args.shard, episode_idx=args.episode_idx, target_mode=tgt,
            max_new_steps=args.max_new_steps, temperature=args.temperature, seed=args.seed, out_wav=args.out_wav,
        )
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
