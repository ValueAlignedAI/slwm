"""SLWM-v3 Live Frozen-Weight Any-to-Any Multimodal Streaming Inference & Media Gallery Runner.

Loads a trained SLWM-v3 checkpoint (`*_ckpt.pkl`) plus optional Wave B (`PolicyCommitGate`) and
Wave C (multimodal decoder) checkpoints with strictly frozen weights (`no` online weight updates!)
and runs `O(1)` causal streaming inference (`models/v3/core.py:step_core`) across ALL modalities,
rendering real playable `.wav` audio (`Kyutai Mimi 32-RVQ` decoder) and real viewable `.png`/`.gif`
video (`Wan 2.1 Causal VAE` decoder) side-by-side with Ground Truth and counterfactual interventions:
  1. `text2text`: Live byte-by-byte streaming text/code completion from any prompt.
  2. `audio2text`: Live streaming captioning / transcription from a pre-encoded shard episode.
  3. `video2text`: Live streaming video-to-text description from a pre-encoded shard episode.
  4. `video2audio`: Strictly blind video-only context -> autoregressive 32-RVQ Mimi audio -> `.wav`
     (`gt.wav`, `pred.wav`, and `shuffled_video_cf.wav` causal conditioning check).
  5. `audio2video`: Strictly blind audio-only context -> autoregressive Wan 2.1 video patches ->
     `.png` + animated `.gif` (`GT | SLWM-v3 Pred | Shuffled-Audio CF`).
  6. `action2video`: Embodied robotics Language + Initial Scene + Motor Actions -> autoregressive
     future Wan 2.1 video rollout -> `.png` + animated `.gif` (`GT | SLWM-v3 Pred | Swapped-Action CF`).
  7. `full_gallery`: Run all 6 any-to-any directions in one pass and write `media_gallery_manifest.json`.
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
    apply_policy_commit_gate,
    apply_video_spatial_refiner,
    predict_any2any_heads,
)
from models.v3.core import init_core_state, step_core


def _maybe_refine_video_groups(heads_params: dict[str, Any], patches_g_15_d: np.ndarray) -> np.ndarray:
    """Apply strictly causal intra-frame `3x5` spatial refiner (`dt_s == 0`) if present in Wave C heads."""
    if "video_spatial_refiner" not in heads_params or patches_g_15_d.size == 0:
        return patches_g_15_d
    ref_out = apply_video_spatial_refiner(
        heads_params["video_spatial_refiner"],
        jnp.asarray(patches_g_15_d, dtype=jnp.float32),
        matmul_dtype=jnp.float32,
    )
    return np.asarray(ref_out, dtype=np.float32)


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

    k_aud = int(params["heads"]["head_audio"]["bias"].shape[0])

    @jax.jit
    def _tick(p, s, pkt):
        z, ns, dm = step_core(p["core"], s, pkt, spec)
        preds = predict_any2any_heads(p["heads"], z[None, :, :], spec, n_pred_audio_codebooks=k_aud, matmul_dtype=jnp.float32)
        return z, ns, preds["text_logits"][0, 0], dm["mean_nowrite"][0], dm["mean_prior_error"][0]

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


def _packet_from_shard(spec, d: dict[str, Any], gi: int, *, is_first: bool, action_override: np.ndarray | None = None) -> dict[str, jax.Array]:
    pkt = _empty_packet_b1(spec)
    pkt["modality_id"] = jnp.array([int(d["modality_id"][gi])], dtype=jnp.int32)
    pkt["dt_s"] = jnp.array([float(d["dt_s"][gi])], dtype=jnp.float32)
    pkt["reliability"] = jnp.array([float(d["reliability"][gi])], dtype=jnp.float32)
    pkt["reset_mask"] = jnp.array([is_first], dtype=jnp.bool_)
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
        act_vec = action_override[None, :] if action_override is not None else d["action_feat"][aci : aci + 1]
        pkt["action_feat"] = jnp.asarray(act_vec, dtype=jnp.float32)
        pkt["has_true_action"] = jnp.array([float(d["has_true_action"][gi])], dtype=jnp.float32)
    return pkt


_CODEC_CACHE: dict[str, Any] = {}


def _resolve_torch_device(device: str = "auto") -> str:
    import torch
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def decode_mimi_codes_to_wav(codes_tq: np.ndarray, out_wav: Path, *, device: str = "auto") -> str:
    """Decode causal `int32[T, Q]` (`Q=8` or `Q=32`) Kyutai Mimi RVQ codes -> playable 24 kHz `.wav`."""
    try:
        import torch
        from scipy.io import wavfile
        from transformers import MimiModel
        dev = _resolve_torch_device(device)
        if "mimi" not in _CODEC_CACHE:
            _CODEC_CACHE["mimi"] = MimiModel.from_pretrained("kyutai/mimi").to(dev).eval()
        mimi = _CODEC_CACHE["mimi"]
        with torch.inference_mode():
            c_t = torch.from_numpy(np.clip(codes_tq, 0, 2047).T[None, :, :].astype(np.int64)).to(dev)  # [1, Q, T]
            wav = mimi.decode(c_t).audio_values[0, 0].float().cpu().numpy()
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(out_wav, 24000, np.clip(wav * 32767.0, -32768, 32767).astype(np.int16))
        return str(out_wav)
    except Exception as exc:
        return f"skipped_mimi_decode: {exc!r}"


def decode_wan_patches_to_rgb(patches_tg_15_1024: np.ndarray, *, device: str = "auto") -> np.ndarray | None:
    """Exact bit-inverse of `v3_preencode_av_codecs.py:encode_video_wan_patches` -> `uint8[T_f, 192, 320, 3]` RGB."""
    try:
        import torch
        from diffusers import AutoencoderKLWan
        dev = _resolve_torch_device(device)
        # Use bf16 on CUDA; fp32 on Apple Silicon MPS/CPU for maximum Conv3d numerical fidelity
        dtype = torch.bfloat16 if dev == "cuda" else torch.float32
        if "wan_vae" not in _CODEC_CACHE:
            _CODEC_CACHE["wan_vae"] = AutoencoderKLWan.from_pretrained(
                "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="vae", torch_dtype=dtype
            ).to(dev).eval()
        vae = _CODEC_CACHE["wan_vae"]
        t_g = int(patches_tg_15_1024.shape[0])
        p = torch.from_numpy(patches_tg_15_1024.astype(np.float32)).to(dev, dtype=dtype)
        # [T_g, 15, 1024] -> [T_g, 3, 5, 8, 8, 16] -> [1, 16, T_g, 24, 40]
        lat = p.reshape(t_g, 3, 5, 8, 8, 16).permute(5, 0, 1, 3, 2, 4).contiguous().reshape(1, 16, t_g, 24, 40)
        with torch.inference_mode():
            vid = vae.decode(lat).sample[0]  # [3, T_f, 192, 320] in [-1, 1]
        rgb = ((vid.float().clamp(-1.0, 1.0) + 1.0) * 127.5).permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)
        return rgb
    except Exception:
        return None


def _sample_rvq_codes(
    logits_kv: np.ndarray, *, temperature: float = 0.80, top_p: float = 0.90, rng: np.random.Generator
) -> np.ndarray:
    """Nucleus (`top_p`) + temperature sampling across `K` Kyutai Mimi RVQ codebooks (avoids greedy RVQ buzz!)."""
    k_aud, vocab = logits_kv.shape
    out = np.zeros((k_aud,), dtype=np.int32)
    for q in range(k_aud):
        lg = logits_kv[q].astype(np.float64)
        if temperature <= 1e-4:
            out[q] = int(np.argmax(lg))
            continue
        lg = (lg - np.max(lg)) / max(temperature, 1e-4)
        p = np.exp(lg)
        p /= np.sum(p)
        if 0.0 < top_p < 1.0:
            order = np.argsort(-p)
            cs = np.cumsum(p[order])
            cut = int(np.searchsorted(cs, top_p)) + 1
            keep = order[: max(1, cut)]
            pk = p[keep]
            pk /= np.sum(pk)
            out[q] = int(rng.choice(keep, p=pk))
        else:
            out[q] = int(rng.choice(vocab, p=p))
    return out


def _rgb_psnr(a_u8: np.ndarray, b_u8: np.ndarray) -> float:
    diff = a_u8.astype(np.float64) - b_u8.astype(np.float64)
    mse = float(np.mean(diff * diff))
    if mse <= 1e-9:
        return 60.0
    return float(10.0 * np.log10((255.0 * 255.0) / mse))


def _save_side_by_side_png_and_gif(
    panel_seqs: list[np.ndarray],
    out_png: Path,
    out_gif: Path,
    *,
    headers: list[str] | None = None,
    fps: float = 6.0,
) -> dict[str, str]:
    """Save labeled horizontal RGB comparison strip (`.png` + animated `.gif` + playable `.mp4` via ffmpeg)."""
    out: dict[str, str] = {}
    try:
        import subprocess
        import tempfile
        from PIL import Image, ImageDraw

        n_f = min(len(s) for s in panel_seqs)
        banner_h = 22 if headers else 0
        strips: list[np.ndarray] = []
        for f in range(n_f):
            row = np.concatenate([s[f] for s in panel_seqs], axis=1)
            if banner_h > 0 and headers:
                canvas = np.zeros((row.shape[0] + banner_h, row.shape[1], 3), dtype=np.uint8)
                canvas[banner_h:] = row
                img = Image.fromarray(canvas)
                draw = ImageDraw.Draw(img)
                pw = panel_seqs[0].shape[2]
                for pi, title in enumerate(headers):
                    draw.text((pi * pw + 6, 4), title[:44], fill=(240, 240, 240))
                strips.append(np.asarray(img, dtype=np.uint8))
            else:
                strips.append(row)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(strips[-1]).save(out_png)
        out["side_by_side_png"] = str(out_png)
        imgs = [Image.fromarray(fr) for fr in strips]
        imgs[0].save(out_gif, save_all=True, append_images=imgs[1:], duration=int(1000 / max(fps, 1.0)), loop=0)
        out["side_by_side_gif"] = str(out_gif)

        # Also render playable `.mp4` trying system/Homebrew ffmpeg or imageio-ffmpeg!
        import shutil
        ffmpeg_bin = shutil.which("ffmpeg") or ("/opt/homebrew/bin/ffmpeg" if Path("/opt/homebrew/bin/ffmpeg").is_file() else None)
        if ffmpeg_bin is None:
            try:
                import imageio_ffmpeg
                ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                ffmpeg_bin = None
        mp4_p = out_gif.with_suffix(".mp4")
        if ffmpeg_bin is not None:
            with tempfile.TemporaryDirectory() as td:
                for fi, im in enumerate(imgs):
                    im.save(Path(td) / f"f_{fi:04d}.png")
                for vcodec in ("libx264", "libopenh264", "mpeg4"):
                    r = subprocess.run(
                        [
                            ffmpeg_bin, "-y", "-v", "error", "-framerate", str(max(2, int(round(fps)))),
                            "-i", f"{td}/f_%04d.png", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                            "-c:v", vcodec, "-pix_fmt", "yuv420p", str(mp4_p),
                        ],
                        capture_output=True,
                        check=False,
                    )
                    if r.returncode == 0 and mp4_p.is_file() and mp4_p.stat().st_size > 0:
                        out["side_by_side_mp4"] = str(mp4_p)
                        break
    except Exception as exc:
        out["render_error"] = repr(exc)
    return out


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
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Feed observed multimodal context from a shard episode through `step_core`, then decode target modality live!

    - For `target_mode == "text"`: feeds pre-separator source modality (`audio` or `video+audio`), decodes UTF-8 bytes.
    - For `target_mode == "audio"` (`video2audio`): feeds strictly VIDEO-ONLY packets (`modality_id == 0`, blind to target
      audio!), generates 32-RVQ Mimi codes autoregressively, also runs a condition-shuffled video counterfactual, and
      decodes `gt.wav`, `pred.wav`, and `shuffled_video_cf.wav`!
    """
    spec, params = bundle["spec"], bundle["params"]
    sh = LoadedSparseShard.load(shard_path)
    n_eps = len(sh.episode_slices)
    ep_i = episode_idx % n_eps
    cf_ep_i = (ep_i + max(1, n_eps // 3)) % n_eps
    s0, e0 = sh.episode_slices[ep_i]
    d = sh.data
    mods = d["modality_id"][s0:e0]
    tb_ep = d["text_byte"][s0:e0]
    sep_pos = np.where(tb_ep == BYTE_MOD_SEP)[0]

    k_aud = int(params["heads"]["head_audio"]["bias"].shape[0])

    @jax.jit
    def _tick(p, s, pkt):
        z, ns, dm = step_core(p["core"], s, pkt, spec)
        preds = predict_any2any_heads(p["heads"], z[None, :, :], spec, n_pred_audio_codebooks=k_aud, matmul_dtype=jnp.float32)
        return z, ns, preds["text_logits"][0, 0], preds["audio_logits"][0, 0], dm["mean_nowrite"][0]

    if target_mode == "text":
        if len(sep_pos) > 0:
            ctx_end = int(sep_pos[0]) + 1
            gt_txt = decode_utf8_bytes(tb_ep[ctx_end:])
        else:
            ctx_end = min(len(mods) // 2, 256)
            gt_txt = decode_utf8_bytes(tb_ep[ctx_end:])
        st = init_core_state(spec, 1)
        rng = np.random.default_rng(seed)
        last_t = None
        nowrites: list[float] = []
        for i in range(min(ctx_end, 512)):
            _, st, last_t, _, nw = _tick(params, st, _packet_from_shard(spec, d, s0 + i, is_first=(i == 0)))
            nowrites.append(float(nw))
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
            _, st, last_t, _, nw = _tick(params, st, pkt)
            nowrites.append(float(nw))
        return {
            "mode": f"cross_modal_to_text ({shard_path.name})",
            "episode_idx": int(ep_i),
            "observed_context_ticks": int(min(ctx_end, 512)),
            "epistemic_label": "predicted_free_running_frozen_weights",
            "ground_truth_text": gt_txt[:240],
            "generated_text": decode_utf8_bytes(np.asarray(gen_b, dtype=np.int32)),
            "mean_nowrite": round(float(np.mean(nowrites)) if nowrites else 0.0, 4),
        }

    # `target_mode == "audio"` (`video2audio`): strictly blind to audio during video conditioning!
    aud_temp = temperature if temperature > 1e-4 else 0.80

    def _run_v2a(src_ep_idx: int, sample_seed: int) -> tuple[np.ndarray, int, float]:
        ss, ee = sh.episode_slices[src_ep_idx]
        st_l = init_core_state(spec, 1)
        rng_a = np.random.default_rng(sample_seed)
        last_a_l = None
        nws_l: list[float] = []
        n_vid = 0
        for gi in range(ss, ee):
            if int(d["modality_id"][gi]) != 0:
                continue  # Strictly VIDEO-ONLY conditioning (zero audio leakage!)
            _, st_l, _, last_a_l, nw = _tick(params, st_l, _packet_from_shard(spec, d, gi, is_first=(n_vid == 0)))
            nws_l.append(float(nw))
            n_vid += 1
            if n_vid >= 180:  # 12 video groups (~1.9s) of observed video
                break
        gen_c: list[np.ndarray] = []
        for _ in range(max_new_steps):
            if last_a_l is None:
                codes_k = np.zeros((k_aud,), dtype=np.int32)
            else:
                codes_k = _sample_rvq_codes(np.asarray(last_a_l), temperature=aud_temp, top_p=0.90, rng=rng_a)
            gen_c.append(codes_k)
            full_codes = np.zeros((spec.audio_codebooks,), dtype=np.int32)
            full_codes[: len(codes_k)] = codes_k
            pkt = _empty_packet_b1(spec)
            pkt["modality_id"] = jnp.array([1], dtype=jnp.int32)
            pkt["dt_s"] = jnp.array([0.08], dtype=jnp.float32)
            pkt["audio_codes"] = jnp.asarray(full_codes[None, :], dtype=jnp.int32)
            _, st_l, _, last_a_l, nw = _tick(params, st_l, pkt)
            nws_l.append(float(nw))
        return np.stack(gen_c, axis=0), n_vid, float(np.mean(nws_l)) if nws_l else 0.0

    # Use identical RNG seed for paired vs shuffled-video so ANY code difference is purely causal video conditioning!
    pred_codes, n_vid_ctx, mean_nw = _run_v2a(ep_i, seed + ep_i)
    cf_codes, _, _ = _run_v2a(cf_ep_i, seed + ep_i)
    # Extract ground-truth Mimi audio codes from the target episode (both matched K_rvq and full 32-RVQ!)
    gt_aud_idx = [int(d["audio_idx"][gi]) for gi in range(s0, e0) if int(d["modality_id"][gi]) == 1 and int(d["audio_idx"][gi]) >= 0]
    gt_codes_32 = d["audio_codes"][gt_aud_idx[:max_new_steps]].astype(np.int32) if gt_aud_idx else np.zeros((len(pred_codes), spec.audio_codebooks), dtype=np.int32)
    gt_codes_k = gt_codes_32[:, :k_aud]
    gt_label = decode_utf8_bytes(tb_ep[int(sep_pos[0]) + 1 :]) if len(sep_pos) > 0 else ""

    # Causal conditioning diagnostic: do predicted RVQ codes change when conditioned on a DIFFERENT video?
    cf_code_diff_rate = float(np.mean(pred_codes != cf_codes))
    rendered: dict[str, str] = {}
    save_root = out_wav.parent if out_wav is not None else out_dir
    if save_root is not None:
        save_root.mkdir(parents=True, exist_ok=True)
        pred_wav_p = out_wav if out_wav is not None else (save_root / f"video2audio_ep{ep_i:04d}_pred_{k_aud}rvq.wav")
        gt_k_wav_p = save_root / f"video2audio_ep{ep_i:04d}_gt_{k_aud}rvq.wav"
        gt_32_wav_p = save_root / f"video2audio_ep{ep_i:04d}_gt_32rvq.wav"
        cf_wav_p = save_root / f"video2audio_ep{ep_i:04d}_shuffled_video_cf_{k_aud}rvq.wav"
        np.savez_compressed(
            save_root / f"video2audio_ep{ep_i:04d}_mimi_codes.npz",
            gt_codes_32=gt_codes_32, gt_codes_k=gt_codes_k, pred_codes=pred_codes, cf_codes=cf_codes,
        )
        rendered["gt_32rvq_wav"] = decode_mimi_codes_to_wav(gt_codes_32, gt_32_wav_p)
        rendered[f"gt_{k_aud}rvq_wav"] = decode_mimi_codes_to_wav(gt_codes_k, gt_k_wav_p)
        rendered[f"pred_{k_aud}rvq_wav"] = decode_mimi_codes_to_wav(pred_codes, pred_wav_p)
        rendered[f"shuffled_video_cf_{k_aud}rvq_wav"] = decode_mimi_codes_to_wav(cf_codes, cf_wav_p)

    return {
        "mode": f"video2audio ({shard_path.name})",
        "episode_idx": int(ep_i),
        "counterfactual_episode_idx": int(cf_ep_i),
        "ground_truth_label": gt_label[:160],
        "observed_video_patches": int(n_vid_ctx),
        "generated_mimi_ticks_12p5hz": int(len(pred_codes)),
        "generated_duration_sec": round(len(pred_codes) / 12.5, 2),
        "causal_video_conditioning_code_diff_rate": round(cf_code_diff_rate, 4),
        "mean_nowrite": round(mean_nw, 4),
        "epistemic_labels": {
            "gt_wav": "observed_ground_truth",
            "pred_wav": "predicted_free_running_frozen_weights",
            "shuffled_video_cf_wav": "counterfactual_shuffled_video_context",
        },
        "artifacts": rendered,
    }


def stream_video_generation_from_shard(
    bundle: dict[str, Any],
    shard_path: Path,
    *,
    episode_idx: int = 0,
    mode: str = "action2video",
    n_pred_groups: int = 3,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Generate Wan 2.1 video frames (`n_pred_groups x 15` patches) from audio (`audio2video`) or scene+action (`action2video`)!

    Always pairs Ground Truth, SLWM-v3 Free-Running Prediction, and a Causal Counterfactual (`Shuffled-Audio CF` for
    `audio2video`; `Swapped/Inverted-Action CF` for `action2video`) decoded to synchronized `.png` + animated `.gif`!
    """
    spec, params = bundle["spec"], bundle["params"]
    sh = LoadedSparseShard.load(shard_path)
    n_eps = len(sh.episode_slices)
    ep_i = episode_idx % n_eps
    cf_ep_i = (ep_i + max(1, n_eps // 3)) % n_eps
    s0, e0 = sh.episode_slices[ep_i]
    d = sh.data
    mods = d["modality_id"][s0:e0]
    vid_pos = np.where(mods == 0)[0]
    n_groups_total = len(vid_pos) // 15
    if n_groups_total < 2:
        return {"error": f"Episode {ep_i} in {shard_path.name} has < 2 video groups ({len(vid_pos)} patches)."}

    k_aud = int(params["heads"]["head_audio"]["bias"].shape[0])

    @jax.jit
    def _tick(p, s, pkt):
        z, ns, dm = step_core(p["core"], s, pkt, spec)
        preds = predict_any2any_heads(p["heads"], z[None, :, :], spec, n_pred_audio_codebooks=k_aud, matmul_dtype=jnp.float32)
        return z, ns, preds["video_pred"][0, 0], preds["action_pred"][0, 0], dm["mean_nowrite"][0]

    if mode == "audio2video":
        use_g = min(n_pred_groups, n_groups_total)
        gt_patches = np.zeros((use_g, 15, spec.video_patch_dim), dtype=np.float32)
        for g in range(use_g):
            for p_idx in range(15):
                vi = int(d["video_idx"][s0 + int(vid_pos[g * 15 + p_idx])])
                if vi >= 0:
                    gt_patches[g, p_idx] = d["video_feat"][vi].astype(np.float32)

        def _run_a2v(audio_ep_idx: int) -> tuple[np.ndarray, int, float]:
            ss, ee = sh.episode_slices[audio_ep_idx]
            aud_gis = [gi for gi in range(ss, ee) if int(d["modality_id"][gi]) == 1]
            st_l = init_core_state(spec, 1)
            last_v_l = None
            nws_l: list[float] = []
            # Strictly AUDIO-ONLY prefix (skip all video patches -> zero visual leakage!)
            # Reserve `use_g - 1` interleaved audio packets to stream between generated video groups (matching AV cadence!).
            n_pre = max(1, min(48, len(aud_gis) - max(0, use_g - 1)))
            for idx_a in range(n_pre):
                _, st_l, last_v_l, _, nw = _tick(
                    params, st_l, _packet_from_shard(spec, d, aud_gis[idx_a], is_first=(idx_a == 0))
                )
                nws_l.append(float(nw))
            pred_p = np.zeros((use_g, 15, spec.video_patch_dim), dtype=np.float32)
            a_cur = n_pre
            for g in range(use_g):
                if g > 0 and a_cur < len(aud_gis):
                    _, st_l, last_v_l, _, nw = _tick(
                        params, st_l, _packet_from_shard(spec, d, aud_gis[a_cur], is_first=False)
                    )
                    nws_l.append(float(nw))
                    a_cur += 1
                for p_idx in range(15):
                    pv = np.asarray(last_v_l, dtype=np.float32) if last_v_l is not None else np.zeros((spec.video_patch_dim,), dtype=np.float32)
                    pred_p[g, p_idx] = pv
                    pkt = _empty_packet_b1(spec)
                    pkt["modality_id"] = jnp.array([0], dtype=jnp.int32)
                    pkt["dt_s"] = jnp.array([0.08 if p_idx == 0 else 0.0], dtype=jnp.float32)
                    pkt["spatial_pos"] = jnp.array([p_idx], dtype=jnp.int32)
                    pkt["video_feat"] = jnp.asarray(pv[None, :], dtype=jnp.float32)
                    _, st_l, last_v_l, _, nw = _tick(params, st_l, pkt)
                    nws_l.append(float(nw))
            return pred_p, a_cur, float(np.mean(nws_l)) if nws_l else 0.0

        pred_patches, n_ctx, mean_nw = _run_a2v(ep_i)
        cf_patches, _, _ = _run_a2v(cf_ep_i)
        pred_patches = _maybe_refine_video_groups(params["heads"], pred_patches)
        cf_patches = _maybe_refine_video_groups(params["heads"], cf_patches)
        cf_label = "counterfactual_shuffled_audio_context"
    else:
        # `action2video`: observe Language + initial `ctx_groups` video + proprio/actions, then autoregressively roll out
        # future video groups along the EXACT interleaved action schedule (feeding ONLY commanded motor actions + the
        # model's OWN predicted proprioception — strictly ZERO future ground-truth video or proprioception leakage!).
        ctx_groups = max(1, min(2, n_groups_total - 1))
        use_g = min(n_pred_groups, n_groups_total - ctx_groups)
        gt_patches = np.zeros((use_g, 15, spec.video_patch_dim), dtype=np.float32)
        for g in range(use_g):
            for p_idx in range(15):
                vi = int(d["video_idx"][s0 + int(vid_pos[(ctx_groups + g) * 15 + p_idx])])
                if vi >= 0:
                    gt_patches[g, p_idx] = d["video_feat"][vi].astype(np.float32)

        start_rel = int(vid_pos[ctx_groups * 15])
        end_rel = int(vid_pos[(ctx_groups + use_g) * 15 - 1]) + 1

        def _run_act2v(*, invert_actions: bool) -> tuple[np.ndarray, int, float]:
            st_l = init_core_state(spec, 1)
            last_v_l = None
            last_act_l = None
            nws_l: list[float] = []
            for rel in range(start_rel):
                _, st_l, last_v_l, last_act_l, nw = _tick(
                    params, st_l, _packet_from_shard(spec, d, s0 + rel, is_first=(rel == 0))
                )
                nws_l.append(float(nw))
            pred_p = np.zeros((use_g, 15, spec.video_patch_dim), dtype=np.float32)
            v_cur = 0
            for rel in range(start_rel, end_rel):
                gi = s0 + rel
                m_id = int(mods[rel])
                if m_id == 0:
                    g, p_idx = v_cur // 15, v_cur % 15
                    pv = np.asarray(last_v_l, dtype=np.float32) if last_v_l is not None else np.zeros((spec.video_patch_dim,), dtype=np.float32)
                    pred_p[g, p_idx] = pv
                    pkt = _packet_from_shard(spec, d, gi, is_first=False)
                    pkt["video_feat"] = jnp.asarray(pv[None, :], dtype=jnp.float32)  # Strictly free-running!
                    _, st_l, last_v_l, last_act_l, nw = _tick(params, st_l, pkt)
                    nws_l.append(float(nw))
                    v_cur += 1
                elif m_id == 3:
                    aci = int(d["act_idx"][gi])
                    a_ov = (-d["action_feat"][aci].astype(np.float32)) if (invert_actions and aci >= 0) else None
                    pkt = _packet_from_shard(spec, d, gi, is_first=False, action_override=a_ov)
                    # Replace ground-truth future proprioception with model's OWN predicted proprio (zero future leakage!)
                    if last_act_l is not None:
                        pred_prop = np.asarray(last_act_l, dtype=np.float32)[spec.action_dim : spec.action_dim + spec.proprio_dim]
                        pkt["proprio_feat"] = jnp.asarray(pred_prop[None, :], dtype=jnp.float32)
                    else:
                        pkt["proprio_feat"] = jnp.zeros((1, spec.proprio_dim), dtype=jnp.float32)
                    _, st_l, last_v_l, last_act_l, nw = _tick(params, st_l, pkt)
                    nws_l.append(float(nw))
                else:
                    _, st_l, last_v_l, last_act_l, nw = _tick(params, st_l, _packet_from_shard(spec, d, gi, is_first=False))
                    nws_l.append(float(nw))
            return pred_p, start_rel, float(np.mean(nws_l)) if nws_l else 0.0

        pred_patches, n_ctx, mean_nw = _run_act2v(invert_actions=False)
        cf_patches, _, _ = _run_act2v(invert_actions=True)
        pred_patches = _maybe_refine_video_groups(params["heads"], pred_patches)
        cf_patches = _maybe_refine_video_groups(params["heads"], cf_patches)
        cf_label = "counterfactual_inverted_motor_action"

    diff = pred_patches - gt_patches
    ad = np.abs(diff)
    huber = float(np.mean(np.where(ad <= 1.0, 0.5 * diff * diff, ad - 0.5)))
    cos = float(np.sum(pred_patches * gt_patches) / (np.linalg.norm(pred_patches) * np.linalg.norm(gt_patches) + 1e-8))
    cf_l2 = float(np.sqrt(np.mean((pred_patches - cf_patches) ** 2)))

    rendered: dict[str, Any] = {}
    rgb_metrics: dict[str, float] = {}
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / f"{mode}_ep{ep_i:04d}_latents.npz",
            gt_patches=gt_patches, pred_patches=pred_patches, cf_patches=cf_patches,
        )
        gt_rgb = decode_wan_patches_to_rgb(gt_patches)
        pr_rgb = decode_wan_patches_to_rgb(pred_patches)
        cf_rgb = decode_wan_patches_to_rgb(cf_patches)
        if gt_rgb is not None and pr_rgb is not None and cf_rgb is not None:
            rgb_metrics["pred_vs_gt_psnr_db"] = round(_rgb_psnr(pr_rgb, gt_rgb), 2)
            rgb_metrics["cf_vs_gt_psnr_db"] = round(_rgb_psnr(cf_rgb, gt_rgb), 2)
            rgb_metrics["pred_vs_cf_psnr_db"] = round(_rgb_psnr(pr_rgb, cf_rgb), 2)
            panel_hdrs = [
                "OBSERVED GROUND TRUTH",
                f"SLWM-v3 {mode.upper()} PRED",
                "SHUFFLED AUDIO CF" if mode == "audio2video" else "INVERTED ACTION CF",
            ]
            rendered.update(
                _save_side_by_side_png_and_gif(
                    [gt_rgb, pr_rgb, cf_rgb],
                    out_dir / f"{mode}_ep{ep_i:04d}_gt_vs_pred_vs_cf.png",
                    out_dir / f"{mode}_ep{ep_i:04d}_gt_vs_pred_vs_cf.gif",
                    headers=panel_hdrs,
                )
            )
            rendered["decoded_frames_shape"] = list(pr_rgb.shape)
            rendered["panel_order"] = panel_hdrs
        else:
            rendered["side_by_side_png"] = "latents_saved_decode_requires_diffusers_wan_vae"

    return {
        "mode": f"{mode} ({shard_path.name})",
        "episode_idx": int(ep_i),
        "observed_context_ticks": int(n_ctx),
        "predicted_video_groups": int(use_g),
        "latent_huber_loss": round(huber, 5),
        "latent_cosine_sim": round(cos, 4),
        "counterfactual_latent_rmse": round(cf_l2, 5),
        "rgb_metrics": rgb_metrics,
        "mean_nowrite": round(mean_nw, 4),
        "epistemic_labels": {
            "left_panel": "observed_ground_truth",
            "middle_panel": "predicted_free_running_frozen_weights",
            "right_panel": cf_label,
        },
        "artifacts": rendered,
    }


def stream_text_to_audio(
    bundle: dict[str, Any],
    prompt: str,
    *,
    max_new_steps: int = 40,
    temperature: float = 0.80,
    seed: int = 42,
    out_dir: Path | None = None,
    tag: str = "prompt0",
) -> dict[str, Any]:
    """True Text -> Audio (`24 kHz .wav`) free-running generation via frozen `step_core` + Kyutai Mimi."""
    spec, params = bundle["spec"], bundle["params"]
    k_aud = int(params["heads"]["head_audio"]["bias"].shape[0])

    @jax.jit
    def _tick(p, s, pkt):
        z, ns, dm = step_core(p["core"], s, pkt, spec)
        preds = predict_any2any_heads(p["heads"], z[None, :, :], spec, n_pred_audio_codebooks=k_aud, matmul_dtype=jnp.float32)
        return ns, preds["audio_logits"][0, 0], dm["mean_nowrite"][0]

    st = init_core_state(spec, 1)
    rng = np.random.default_rng(seed)
    p_bytes = list(prompt.encode("utf-8", errors="ignore")) + [BYTE_MOD_SEP]
    last_a = None
    nws: list[float] = []
    for i, b_val in enumerate(p_bytes):
        pkt = _empty_packet_b1(spec)
        pkt["modality_id"] = jnp.array([2], dtype=jnp.int32)
        pkt["text_byte"] = jnp.array([b_val], dtype=jnp.int32)
        pkt["reset_mask"] = jnp.array([i == 0], dtype=jnp.bool_)
        st, last_a, nw = _tick(params, st, pkt)
        nws.append(float(nw))

    gen_c: list[np.ndarray] = []
    for _ in range(max_new_steps):
        codes_k = _sample_rvq_codes(np.asarray(last_a), temperature=max(temperature, 0.75), top_p=0.90, rng=rng)
        gen_c.append(codes_k)
        full_c = np.zeros((spec.audio_codebooks,), dtype=np.int32)
        full_c[: len(codes_k)] = codes_k
        pkt = _empty_packet_b1(spec)
        pkt["modality_id"] = jnp.array([1], dtype=jnp.int32)
        pkt["dt_s"] = jnp.array([0.08], dtype=jnp.float32)
        pkt["audio_codes"] = jnp.asarray(full_c[None, :], dtype=jnp.int32)
        st, last_a, nw = _tick(params, st, pkt)
        nws.append(float(nw))
    codes_arr = np.stack(gen_c, axis=0)
    rendered: dict[str, str] = {}
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / f"text2audio_{tag}_mimi_codes.npz", pred_codes=codes_arr)
        rendered["pred_wav"] = decode_mimi_codes_to_wav(codes_arr, out_dir / f"text2audio_{tag}_pred_{k_aud}rvq.wav")
    return {
        "mode": "text2audio",
        "prompt": prompt,
        "generated_duration_sec": round(len(codes_arr) / 12.5, 2),
        "mean_nowrite": round(float(np.mean(nws)) if nws else 0.0, 4),
        "epistemic_label": "imagined_text_conditioned_audio_frozen_weights",
        "artifacts": rendered,
    }


def stream_text_to_video(
    bundle: dict[str, Any],
    prompt: str,
    *,
    n_pred_groups: int = 3,
    out_dir: Path | None = None,
    tag: str = "prompt0",
) -> dict[str, Any]:
    """True Text -> Video (`192x320` RGB `.mp4`/`.gif`/`.png`) free-running generation via frozen `step_core` + Wan 2.1 VAE."""
    spec, params = bundle["spec"], bundle["params"]
    k_aud = int(params["heads"]["head_audio"]["bias"].shape[0])

    @jax.jit
    def _tick(p, s, pkt):
        z, ns, dm = step_core(p["core"], s, pkt, spec)
        preds = predict_any2any_heads(p["heads"], z[None, :, :], spec, n_pred_audio_codebooks=k_aud, matmul_dtype=jnp.float32)
        return ns, preds["video_pred"][0, 0], dm["mean_nowrite"][0]

    st = init_core_state(spec, 1)
    p_bytes = list(prompt.encode("utf-8", errors="ignore")) + [BYTE_MOD_SEP]
    last_v = None
    nws: list[float] = []
    for i, b_val in enumerate(p_bytes):
        pkt = _empty_packet_b1(spec)
        pkt["modality_id"] = jnp.array([2], dtype=jnp.int32)
        pkt["text_byte"] = jnp.array([b_val], dtype=jnp.int32)
        pkt["reset_mask"] = jnp.array([i == 0], dtype=jnp.bool_)
        st, last_v, nw = _tick(params, st, pkt)
        nws.append(float(nw))

    pred_p = np.zeros((n_pred_groups, 15, spec.video_patch_dim), dtype=np.float32)
    for g in range(n_pred_groups):
        for p_idx in range(15):
            pv = np.asarray(last_v, dtype=np.float32) if last_v is not None else np.zeros((spec.video_patch_dim,), dtype=np.float32)
            pred_p[g, p_idx] = pv
            pkt = _empty_packet_b1(spec)
            pkt["modality_id"] = jnp.array([0], dtype=jnp.int32)
            pkt["dt_s"] = jnp.array([0.0 if p_idx > 0 else 0.16], dtype=jnp.float32)
            pkt["spatial_pos"] = jnp.array([p_idx], dtype=jnp.int32)
            pkt["video_feat"] = jnp.asarray(pv[None, :], dtype=jnp.float32)
            st, last_v, nw = _tick(params, st, pkt)
            nws.append(float(nw))

    pred_p = _maybe_refine_video_groups(params["heads"], pred_p)
    rendered: dict[str, Any] = {}
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / f"text2video_{tag}_latents.npz", pred_patches=pred_p)
        pr_rgb = decode_wan_patches_to_rgb(pred_p)
        if pr_rgb is not None:
            rendered.update(
                _save_side_by_side_png_and_gif(
                    [pr_rgb],
                    out_dir / f"text2video_{tag}_pred.png",
                    out_dir / f"text2video_{tag}_pred.gif",
                    headers=[f"TEXT2VIDEO: {prompt[:30]}"],
                )
            )
            rendered["decoded_frames_shape"] = list(pr_rgb.shape)
    return {
        "mode": "text2video",
        "prompt": prompt,
        "predicted_video_groups": int(n_pred_groups),
        "mean_nowrite": round(float(np.mean(nws)) if nws else 0.0, 4),
        "epistemic_label": "imagined_text_conditioned_video_frozen_weights",
        "artifacts": rendered,
    }


def stream_audio_continuation_from_shard(
    bundle: dict[str, Any],
    shard_path: Path,
    *,
    episode_idx: int = 0,
    ctx_ticks: int = 25,
    gen_ticks: int = 32,
    temperature: float = 0.80,
    seed: int = 42,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """True Audio -> Audio (`24 kHz .wav`) continuation: observe `ctx_ticks` (`2.0s`), autoregressively continue `gen_ticks` (`2.56s`)."""
    spec, params = bundle["spec"], bundle["params"]
    k_aud = int(params["heads"]["head_audio"]["bias"].shape[0])
    sh = LoadedSparseShard.load(shard_path)
    ep_i = episode_idx % len(sh.episode_slices)
    s0, e0 = sh.episode_slices[ep_i]
    d = sh.data
    aud_gis = [gi for gi in range(s0, e0) if int(d["modality_id"][gi]) == 1 and int(d["audio_idx"][gi]) >= 0]
    if len(aud_gis) < ctx_ticks + 8:
        return {"error": f"Episode {ep_i} has only {len(aud_gis)} audio ticks."}

    @jax.jit
    def _tick(p, s, pkt):
        z, ns, dm = step_core(p["core"], s, pkt, spec)
        preds = predict_any2any_heads(p["heads"], z[None, :, :], spec, n_pred_audio_codebooks=k_aud, matmul_dtype=jnp.float32)
        return ns, preds["audio_logits"][0, 0], dm["mean_nowrite"][0]

    st = init_core_state(spec, 1)
    rng = np.random.default_rng(seed + ep_i)
    last_a = None
    nws: list[float] = []
    ctx_codes_k = []
    for idx_i, gi in enumerate(aud_gis[:ctx_ticks]):
        ai = int(d["audio_idx"][gi])
        ctx_codes_k.append(d["audio_codes"][ai, :k_aud].astype(np.int32))
        st, last_a, nw = _tick(params, st, _packet_from_shard(spec, d, gi, is_first=(idx_i == 0)))
        nws.append(float(nw))

    gen_c: list[np.ndarray] = []
    for _ in range(gen_ticks):
        codes_k = _sample_rvq_codes(np.asarray(last_a), temperature=max(temperature, 0.75), top_p=0.90, rng=rng)
        gen_c.append(codes_k)
        full_c = np.zeros((spec.audio_codebooks,), dtype=np.int32)
        full_c[: len(codes_k)] = codes_k
        pkt = _empty_packet_b1(spec)
        pkt["modality_id"] = jnp.array([1], dtype=jnp.int32)
        pkt["dt_s"] = jnp.array([0.08], dtype=jnp.float32)
        pkt["audio_codes"] = jnp.asarray(full_c[None, :], dtype=jnp.int32)
        st, last_a, nw = _tick(params, st, pkt)
        nws.append(float(nw))

    gt_all_k = np.stack([d["audio_codes"][int(d["audio_idx"][gi]), :k_aud].astype(np.int32) for gi in aud_gis[: ctx_ticks + gen_ticks]], axis=0)
    pred_full_k = np.concatenate([np.stack(ctx_codes_k, axis=0), np.stack(gen_c, axis=0)], axis=0)
    rendered: dict[str, str] = {}
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / f"audio2audio_ep{ep_i:04d}_mimi_codes.npz", gt_codes=gt_all_k, pred_codes=pred_full_k)
        rendered["gt_full_wav"] = decode_mimi_codes_to_wav(gt_all_k, out_dir / f"audio2audio_ep{ep_i:04d}_gt_{k_aud}rvq.wav")
        rendered["pred_continuation_wav"] = decode_mimi_codes_to_wav(pred_full_k, out_dir / f"audio2audio_ep{ep_i:04d}_pred_cont_{k_aud}rvq.wav")
    return {
        "mode": f"audio2audio ({shard_path.name})",
        "episode_idx": int(ep_i),
        "observed_prefix_sec": round(ctx_ticks / 12.5, 2),
        "generated_continuation_sec": round(gen_ticks / 12.5, 2),
        "mean_nowrite": round(float(np.mean(nws)) if nws else 0.0, 4),
        "epistemic_label": "prefix_observed_suffix_predicted_frozen_weights",
        "artifacts": rendered,
    }


def stream_video_to_action_from_shard(
    bundle: dict[str, Any],
    shard_path: Path,
    *,
    episode_idx: int = 0,
    max_steps: int = 24,
) -> dict[str, Any]:
    """True Video+Instruction -> Motor Action prediction (strictly blind to ground-truth motor actions!)."""
    spec, params = bundle["spec"], bundle["params"]
    k_aud = int(params["heads"]["head_audio"]["bias"].shape[0])
    sh = LoadedSparseShard.load(shard_path)
    ep_i = episode_idx % len(sh.episode_slices)
    s0, e0 = sh.episode_slices[ep_i]
    d = sh.data

    @jax.jit
    def _tick(p, s, pkt):
        z, ns, dm = step_core(p["core"], s, pkt, spec)
        preds = predict_any2any_heads(p["heads"], z[None, :, :], spec, n_pred_audio_codebooks=k_aud, matmul_dtype=jnp.float32)
        return ns, preds["action_pred"][0, 0], dm["mean_nowrite"][0]

    st = init_core_state(spec, 1)
    last_ap = None
    gt_acts, pr_acts, nws = [], [], []
    zero_act = np.zeros((spec.action_dim,), dtype=np.float32)
    for rel in range(e0 - s0):
        gi = s0 + rel
        m_id = int(d["modality_id"][gi])
        if m_id == 3 and int(d["act_idx"][gi]) >= 0 and int(d["has_true_action"][gi]) > 0:
            if last_ap is not None:
                gt_acts.append(d["action_feat"][int(d["act_idx"][gi])].astype(np.float32))
                pr_acts.append(np.asarray(last_ap[: spec.action_dim], dtype=np.float32))
                if len(gt_acts) >= max_steps:
                    break
            # Feed proprioception ONLY with zeroed action & has_true_action=0 (blind to GT action!)
            pkt = _packet_from_shard(spec, d, gi, is_first=(rel == 0), action_override=zero_act)
            pkt["has_true_action"] = jnp.array([0.0], dtype=jnp.float32)
        else:
            pkt = _packet_from_shard(spec, d, gi, is_first=(rel == 0))
        st, last_ap, nw = _tick(params, st, pkt)
        nws.append(float(nw))

    if not gt_acts:
        return {"error": f"No motor actions found in episode {ep_i}."}
    g_arr, p_arr = np.stack(gt_acts), np.stack(pr_acts)
    diff = p_arr - g_arr
    ad = np.abs(diff)
    huber = float(np.mean(np.where(ad <= 1.0, 0.5 * diff * diff, ad - 0.5)))
    cos = float(np.sum(p_arr * g_arr) / (np.linalg.norm(p_arr) * np.linalg.norm(g_arr) + 1e-8))
    return {
        "mode": f"video2action ({shard_path.name})",
        "episode_idx": int(ep_i),
        "predicted_action_steps": int(len(gt_acts)),
        "action_huber_loss": round(huber, 5),
        "action_l1_error": round(float(np.mean(ad)), 5),
        "action_cosine_sim": round(cos, 4),
        "mean_nowrite": round(float(np.mean(nws)) if nws else 0.0, 4),
        "epistemic_label": "predicted_motor_action_blind_to_gt_actions",
    }


def run_full_any2any_media_gallery(
    bundle: dict[str, Any],
    shard_dir: Path,
    out_dir: Path,
    *,
    episodes: list[int] = [0, 1],
    non_text_only: bool = False,
) -> dict[str, Any]:
    """Render a complete non-text (and optional text) any-to-any media gallery (`.wav`, `.mp4`, `.gif`, `.png`, motor actions)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    gallery: dict[str, Any] = {
        "frozen_weights_only": True,
        "non_text_only": bool(non_text_only),
        "video2audio_wav": [],
        "audio2video_mp4_gif_png": [],
        "action2video_mp4_gif_png": [],
        "audio2audio_wav": [],
        "text2audio_wav": [],
        "text2video_mp4_gif_png": [],
        "video2action_motor": [],
    }
    if not non_text_only:
        gallery["text2text"] = []
        gallery["audio2text"] = []
        gallery["video2text"] = []
        for pr in [
            "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n",
            "The European Space Agency announced that the new telescope ",
        ]:
            gallery["text2text"].append(stream_text_continuation(bundle, pr, max_new_bytes=120, temperature=0.0))

    clotho = shard_dir / "audio_text/clotho_test.npz"
    if not clotho.is_file():
        clotho = shard_dir / "audio_text/clotho_dev.npz"
    if clotho.is_file():
        for ep in episodes:
            if not non_text_only:
                gallery["audio2text"].append(stream_cross_modal_from_shard(bundle, clotho, episode_idx=ep, target_mode="text", max_new_steps=96))
            gallery["audio2audio_wav"].append(
                stream_audio_continuation_from_shard(bundle, clotho, episode_idx=ep, out_dir=out_dir / "audio")
            )

    msrvtt = shard_dir / "av/msrvtt_test.npz"
    if not msrvtt.is_file():
        msrvtt = shard_dir / "av/msrvtt_dev.npz"
    if msrvtt.is_file() and not non_text_only:
        for ep in episodes:
            gallery["video2text"].append(stream_cross_modal_from_shard(bundle, msrvtt, episode_idx=ep, target_mode="text", max_new_steps=96))

    vgg = shard_dir / "av/vggsound_test.npz"
    if not vgg.is_file():
        vgg = shard_dir / "av/vggsound_dev.npz"
    if vgg.is_file():
        for ep in episodes:
            gallery["video2audio_wav"].append(
                stream_cross_modal_from_shard(bundle, vgg, episode_idx=ep, target_mode="audio", max_new_steps=32, out_dir=out_dir / "audio")
            )
            gallery["audio2video_mp4_gif_png"].append(
                stream_video_generation_from_shard(bundle, vgg, episode_idx=ep, mode="audio2video", n_pred_groups=3, out_dir=out_dir / "video")
            )

    rob = shard_dir / "robotics/robotics_test.npz"
    if not rob.is_file():
        rob = shard_dir / "robotics/robotics_dev.npz"
    if rob.is_file():
        for ep in episodes:
            gallery["action2video_mp4_gif_png"].append(
                stream_video_generation_from_shard(bundle, rob, episode_idx=ep, mode="action2video", n_pred_groups=3, out_dir=out_dir / "video")
            )
            gallery["video2action_motor"].append(stream_video_to_action_from_shard(bundle, rob, episode_idx=ep))

    for ti, pr_a in enumerate(["sound: water rushing in a stream", "sound: dog barking loudly"]):
        gallery["text2audio_wav"].append(stream_text_to_audio(bundle, pr_a, out_dir=out_dir / "audio", tag=f"p{ti}"))
    for ti, pr_v in enumerate(["a car driving down a road", "pick up the object on the table"]):
        gallery["text2video_mp4_gif_png"].append(stream_text_to_video(bundle, pr_v, out_dir=out_dir / "video", tag=f"p{ti}"))

    manifest_p = out_dir / "media_gallery_manifest.json"
    manifest_p.write_text(json.dumps(gallery, indent=2), encoding="utf-8")
    return gallery


def main() -> None:
    ap = argparse.ArgumentParser(description="Live frozen-weight O(1) streaming any-to-any inference & media gallery for SLWM-v3.")
    ap.add_argument("--ckpt", type=Path, required=True, help="Path to saved `*_ckpt.pkl` core+heads checkpoint.")
    ap.add_argument("--wave-b-ckpt", type=Path, default=None, help="Optional saved `wave_b_*_ckpt.pkl` PolicyCommitGate.")
    ap.add_argument("--wave-c-ckpt", type=Path, default=None, help="Optional saved `wave_c_*_ckpt.pkl` Wave C decoders.")
    ap.add_argument(
        "--mode",
        type=str,
        default="non_text_gallery",
        choices=(
            "text2text", "audio2text", "video2text",
            "video2audio", "audio2video", "action2video",
            "audio2audio", "text2audio", "text2video", "video2action",
            "non_text_gallery", "full_gallery",
        ),
    )
    ap.add_argument("--prompt", type=str, default="sound: water rushing in a stream")
    ap.add_argument("--shard", type=Path, default=None, help="Pre-encoded shard `.npz` for cross-modal modes.")
    ap.add_argument("--shard-dir", type=Path, default=Path("/mnt/slwm-data/SLWM/v3_preencoded"), help="Shard root for gallery modes.")
    ap.add_argument("--episode-idx", type=int, default=0)
    ap.add_argument("--max-new-steps", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out-wav", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("reports/v3/live_media_showcase"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    bundle = load_frozen_bundle(args.ckpt, wave_b_ckpt=args.wave_b_ckpt, wave_c_ckpt=args.wave_c_ckpt)
    if args.mode in ("full_gallery", "non_text_gallery"):
        res = run_full_any2any_media_gallery(
            bundle, args.shard_dir, args.out_dir,
            episodes=[args.episode_idx, args.episode_idx + 1],
            non_text_only=(args.mode == "non_text_gallery"),
        )
    elif args.mode == "text2text":
        res = stream_text_continuation(
            bundle, args.prompt, max_new_bytes=args.max_new_steps, temperature=args.temperature, seed=args.seed
        )
    elif args.mode == "text2audio":
        res = stream_text_to_audio(bundle, args.prompt, max_new_steps=min(64, args.max_new_steps), temperature=args.temperature, seed=args.seed, out_dir=args.out_dir / "audio")
    elif args.mode == "text2video":
        res = stream_text_to_video(bundle, args.prompt, out_dir=args.out_dir / "video")
    elif args.mode == "audio2audio":
        if args.shard is None:
            raise ValueError("--shard is required for mode=audio2audio")
        res = stream_audio_continuation_from_shard(bundle, args.shard, episode_idx=args.episode_idx, temperature=args.temperature, seed=args.seed, out_dir=args.out_dir / "audio")
    elif args.mode == "video2action":
        if args.shard is None:
            raise ValueError("--shard is required for mode=video2action")
        res = stream_video_to_action_from_shard(bundle, args.shard, episode_idx=args.episode_idx)
    elif args.mode in ("audio2video", "action2video"):
        if args.shard is None:
            raise ValueError(f"--shard is required for mode={args.mode}")
        res = stream_video_generation_from_shard(
            bundle, args.shard, episode_idx=args.episode_idx, mode=args.mode, out_dir=args.out_dir
        )
    else:
        if args.shard is None:
            raise ValueError(f"--shard is required for mode={args.mode}")
        tgt = "audio" if args.mode == "video2audio" else "text"
        res = stream_cross_modal_from_shard(
            bundle, args.shard, episode_idx=args.episode_idx, target_mode=tgt,
            max_new_steps=args.max_new_steps, temperature=args.temperature, seed=args.seed,
            out_wav=args.out_wav, out_dir=args.out_dir,
        )
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
