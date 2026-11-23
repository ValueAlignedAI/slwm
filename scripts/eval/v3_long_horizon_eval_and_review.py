"""SLWM-v3 Long-Horizon (>1.0s) Multi-Model Any-to-Any Generation & Quantitative Artifact Audit.

Runs strictly frozen-weight O(1) streaming rollouts longer than 1 second (`1.8s–4.0s` video and
`4.0s` 24 kHz audio, plus `2.4s` blind motor control and `256`-byte text/code/captioning) across
converged `v2` checkpoints, renders playable `.mp4`/`.gif`/`.wav` artifacts + temporal filmstrips +
log-mel spectrogram comparison figures, and computes objective DSP, vision, temporal-drift, and
causal-counterfactual metrics for every generated artifact.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.v3_unified_shards import BYTE_EOS, BYTE_MOD_SEP, LoadedSparseShard, decode_utf8_bytes
from models.v3.any2any_heads import predict_any2any_heads
from models.v3.core import init_core_state, step_core
from scripts.eval.v3_live_any2any_inference import (
    _empty_packet_b1,
    _maybe_refine_video_groups,
    _packet_from_shard,
    _rgb_psnr,
    _sample_rvq_codes,
    _save_side_by_side_png_and_gif,
    decode_mimi_codes_to_wav,
    decode_wan_patches_to_rgb,
    load_frozen_bundle,
    stream_audio_continuation_from_shard,
    stream_cross_modal_from_shard,
    stream_text_continuation,
    stream_text_to_audio,
    stream_text_to_video,
    stream_video_generation_from_shard,
    stream_video_to_action_from_shard,
)


# ---------------------------------------------------------------------------
# Objective Audio DSP & Spectrogram Evaluation Helpers
# ---------------------------------------------------------------------------


def _load_wav_float(wav_path: Path) -> tuple[int, np.ndarray] | None:
    if not wav_path.is_file():
        return None
    from scipy.io import wavfile

    sr, x = wavfile.read(wav_path)
    if x.ndim > 1:
        x = x.mean(axis=-1)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / float(max(abs(np.iinfo(x.dtype).min), 1))
    else:
        x = x.astype(np.float32)
    return int(sr), x


def _stft_log_mel(x: np.ndarray, sr: int = 24000, n_fft: int = 1024, hop: int = 256, n_mels: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute log-mel spectrogram `[n_mels, T]`, power spectrogram `[F, T]`, and freq bins `[F]`."""
    from scipy.signal import stft

    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    freqs, _, zxx = stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary="zeros")
    pwr = np.abs(zxx) ** 2 + 1e-10  # [F, T]
    # Triangular Mel filterbank
    def _hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def _mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    m_pts = np.linspace(_hz_to_mel(50.0), _hz_to_mel(sr / 2.0), n_mels + 2)
    hz_pts = _mel_to_hz(m_pts)
    fb = np.zeros((n_mels, len(freqs)), dtype=np.float32)
    for i in range(n_mels):
        lo, mid, hi = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        up = (freqs - lo) / max(mid - lo, 1e-6)
        dn = (hi - freqs) / max(hi - mid, 1e-6)
        fb[i] = np.maximum(0.0, np.minimum(up, dn))
    mel = np.maximum(fb @ pwr, 1e-10)
    log_mel = np.log10(mel)
    return log_mel.astype(np.float32), pwr.astype(np.float32), freqs.astype(np.float32)


def _audio_dsp_metrics(wav_path: Path, ref_wav_path: Path | None = None) -> dict[str, Any]:
    loaded = _load_wav_float(wav_path)
    if loaded is None:
        return {"error": f"missing_wav: {wav_path}"}
    sr, x = loaded
    dur_s = len(x) / float(sr)
    rms = float(np.sqrt(np.mean(x * x) + 1e-12))
    peak = float(np.max(np.abs(x)) + 1e-12)
    zcr = float(np.mean(np.abs(np.diff(np.signbit(x)))))
    lm, pwr, freqs = _stft_log_mel(x, sr=sr)
    # Spectral centroid (Hz) & Wiener spectral flatness (0=pure tone/buzz, 1=white noise)
    mag = np.sqrt(pwr)
    cent = float(np.mean(np.sum(freqs[:, None] * mag, axis=0) / np.maximum(np.sum(mag, axis=0), 1e-8)))
    flat = float(np.mean(np.exp(np.mean(np.log(pwr), axis=0)) / np.maximum(np.mean(pwr, axis=0), 1e-10)))
    # Framewise energy envelope (for burst/impact dynamics)
    env = np.sqrt(np.mean(pwr, axis=0))
    env_cv = float(np.std(env) / max(float(np.mean(env)), 1e-8))
    # Temporal self-similarity (lag-1 frame cosine: >0.99 indicates frozen static drone/buzz)
    lm_n = lm / (np.linalg.norm(lm, axis=0, keepdims=True) + 1e-8)
    lag1_cos = float(np.mean(np.sum(lm_n[:, :-1] * lm_n[:, 1:], axis=0))) if lm.shape[1] > 2 else 0.0
    out: dict[str, Any] = {
        "duration_s": round(dur_s, 3),
        "rms": round(rms, 5),
        "peak": round(peak, 4),
        "zcr": round(zcr, 4),
        "spectral_centroid_hz": round(cent, 1),
        "spectral_flatness": round(flat, 4),
        "envelope_cv": round(env_cv, 4),
        "lag1_mel_cos": round(lag1_cos, 4),
    }
    if ref_wav_path is not None:
        ref = _load_wav_float(ref_wav_path)
        if ref is not None:
            _, xr = ref
            lm_r, pwr_r, _ = _stft_log_mel(xr, sr=sr)
            t_min = min(lm.shape[1], lm_r.shape[1])
            out["log_mel_l1_vs_ref"] = round(float(np.mean(np.abs(lm[:, :t_min] - lm_r[:, :t_min]))), 4)
            # Cosine similarity of mean spectral profile
            mp, mr = np.mean(lm[:, :t_min], axis=1), np.mean(lm_r[:, :t_min], axis=1)
            out["mean_mel_cos_vs_ref"] = round(float(np.dot(mp, mr) / (np.linalg.norm(mp) * np.linalg.norm(mr) + 1e-8)), 4)
            # Energy envelope correlation (detects temporal impact alignment on GreatestHits!)
            e_p = np.sqrt(np.mean(pwr[:, :t_min], axis=0))
            e_r = np.sqrt(np.mean(pwr_r[:, :t_min], axis=0))
            if np.std(e_p) > 1e-8 and np.std(e_r) > 1e-8:
                out["envelope_pearson_vs_ref"] = round(float(np.corrcoef(e_p, e_r)[0, 1]), 4)
            else:
                out["envelope_pearson_vs_ref"] = 0.0
    return out


def _rvq_code_diversity(codes_tq: np.ndarray) -> dict[str, float]:
    """Compute RVQ-0 semantic token diversity, entropy, and bigram repetition rate."""
    if codes_tq.size == 0:
        return {"rvq0_unique_ratio": 0.0, "rvq0_entropy_bits": 0.0, "rvq0_repeat_rate": 0.0}
    c0 = codes_tq[:, 0]
    uniq, cnts = np.unique(c0, return_counts=True)
    probs = cnts.astype(np.float64) / max(len(c0), 1)
    ent = float(-np.sum(probs * np.log2(np.maximum(probs, 1e-12))))
    rep = float(np.mean(c0[1:] == c0[:-1])) if len(c0) > 1 else 0.0
    return {
        "rvq0_unique_ratio": round(len(uniq) / max(len(c0), 1), 4),
        "rvq0_entropy_bits": round(ent, 3),
        "rvq0_adj_repeat_rate": round(rep, 4),
    }


def _save_spectrogram_panel(wav_specs: list[tuple[str, Path]], out_png: Path) -> str:
    """Render stacked/horizontal log-mel spectrograms (`uint8` magma-like palette) for visual inspection."""
    from PIL import Image, ImageDraw

    panels: list[np.ndarray] = []
    titles: list[str] = []
    lms: list[np.ndarray] = []
    for title, wp in wav_specs:
        ld = _load_wav_float(wp)
        if ld is None:
            continue
        sr, x = ld
        lm, _, _ = _stft_log_mel(x, sr=sr)
        lms.append(lm)
        titles.append(title)
    if not lms:
        return ""
    vmin = float(np.percentile(np.concatenate([m.ravel() for m in lms]), 5))
    vmax = float(np.percentile(np.concatenate([m.ravel() for m in lms]), 99))
    span = max(vmax - vmin, 1e-4)
    w_target = 360
    h_target = 140
    for title, lm in zip(titles, lms):
        norm = np.clip((lm[::-1, :] - vmin) / span, 0.0, 1.0)  # high freq on top
        # Clean perceptual inferno-like RGB ramp
        r = np.clip(255.0 * np.minimum(1.0, 1.6 * norm), 0, 255).astype(np.uint8)
        g = np.clip(255.0 * np.maximum(0.0, 1.5 * (norm - 0.30)), 0, 255).astype(np.uint8)
        b = np.clip(255.0 * (0.65 * np.sin(np.pi * norm) + 0.35 * (norm ** 2)), 0, 255).astype(np.uint8)
        rgb = np.stack([r, g, b], axis=-1)
        im = Image.fromarray(rgb).resize((w_target, h_target), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (w_target, h_target + 22), (12, 12, 16))
        canvas.paste(im, (0, 22))
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 4), title[:46], fill=(240, 240, 245))
        panels.append(np.asarray(canvas, dtype=np.uint8))
    grid = np.concatenate(panels, axis=1)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(out_png)
    return str(out_png)


# ---------------------------------------------------------------------------
# Video Temporal Rollout & Filmstrip Evaluation Helpers
# ---------------------------------------------------------------------------


def _eval_video_rollout_curve(
    gt_p: np.ndarray,
    pr_p: np.ndarray,
    cf_p: np.ndarray,
    gt_rgb: np.ndarray,
    pr_rgb: np.ndarray,
    cf_rgb: np.ndarray,
    *,
    fps_assumed: float,
) -> dict[str, Any]:
    """Compute per-group temporal drift curve, pixel motion, SSIM proxy, and causal CF advantage."""
    g_n = gt_p.shape[0]
    n_f = min(len(gt_rgb), len(pr_rgb), len(cf_rgb))
    dur_s = n_f / max(fps_assumed, 1.0)
    per_group = []
    for g in range(g_n):
        d = pr_p[g] - gt_p[g]
        ad = np.abs(d)
        hub = float(np.mean(np.where(ad <= 1.0, 0.5 * d * d, ad - 0.5)))
        cos = float(np.sum(pr_p[g] * gt_p[g]) / (np.linalg.norm(pr_p[g]) * np.linalg.norm(gt_p[g]) + 1e-8))
        cf_rmse = float(np.sqrt(np.mean((pr_p[g] - cf_p[g]) ** 2)))
        norm_ratio = float(np.linalg.norm(pr_p[g]) / (np.linalg.norm(gt_p[g]) + 1e-8))
        f_idx = min(n_f - 1, g * 4)
        psnr_g = _rgb_psnr(pr_rgb[f_idx], gt_rgb[f_idx])
        psnr_cf = _rgb_psnr(cf_rgb[f_idx], gt_rgb[f_idx])
        per_group.append(
            {
                "group": g,
                "t_sec": round(f_idx / max(fps_assumed, 1.0), 2),
                "huber": round(hub, 4),
                "cos": round(cos, 4),
                "norm_ratio": round(norm_ratio, 3),
                "cf_rmse": round(cf_rmse, 4),
                "psnr_db": round(psnr_g, 2),
                "cf_psnr_db": round(psnr_cf, 2),
                "psnr_margin_over_cf_db": round(psnr_g - psnr_cf, 2),
            }
        )
    # Frame-to-frame pixel motion (mean abs RGB diff): diagnoses static freeze!
    gt_motion = float(np.mean(np.abs(np.diff(gt_rgb[:n_f].astype(np.float32), axis=0)))) if n_f > 1 else 0.0
    pr_motion = float(np.mean(np.abs(np.diff(pr_rgb[:n_f].astype(np.float32), axis=0)))) if n_f > 1 else 0.0
    cf_motion = float(np.mean(np.abs(np.diff(cf_rgb[:n_f].astype(np.float32), axis=0)))) if n_f > 1 else 0.0
    # Split early (first third) vs late (last third) PSNR/Huber
    k3 = max(1, g_n // 3)
    early_hub = float(np.mean([x["huber"] for x in per_group[:k3]]))
    late_hub = float(np.mean([x["huber"] for x in per_group[-k3:]]))
    early_psnr = float(np.mean([x["psnr_db"] for x in per_group[:k3]]))
    late_psnr = float(np.mean([x["psnr_db"] for x in per_group[-k3:]]))
    return {
        "n_groups": g_n,
        "n_frames": n_f,
        "duration_s_at_fps": round(dur_s, 2),
        "fps_assumed": fps_assumed,
        "overall_psnr_db": round(_rgb_psnr(pr_rgb[:n_f], gt_rgb[:n_f]), 2),
        "overall_cf_psnr_db": round(_rgb_psnr(cf_rgb[:n_f], gt_rgb[:n_f]), 2),
        "overall_psnr_gain_over_cf_db": round(_rgb_psnr(pr_rgb[:n_f], gt_rgb[:n_f]) - _rgb_psnr(cf_rgb[:n_f], gt_rgb[:n_f]), 2),
        "early_huber": round(early_hub, 4),
        "late_huber": round(late_hub, 4),
        "drift_ratio_late_over_early": round(late_hub / max(early_hub, 1e-6), 2),
        "early_psnr_db": round(early_psnr, 2),
        "late_psnr_db": round(late_psnr, 2),
        "gt_frame_motion_rgb_l1": round(gt_motion, 3),
        "pred_frame_motion_rgb_l1": round(pr_motion, 3),
        "cf_frame_motion_rgb_l1": round(cf_motion, 3),
        "per_group_curve": per_group,
    }


def _save_temporal_filmstrip(
    gt_rgb: np.ndarray,
    pr_rgb: np.ndarray,
    cf_rgb: np.ndarray,
    out_png: Path,
    *,
    title: str,
    fps_assumed: float,
    n_cols: int = 6,
) -> str:
    """Save 4-row x 6-timestamp filmstrip (`GT`, `PRED`, `CF`, `4x |PRED-CF|`) to visually audit >1s evolution."""
    from PIL import Image, ImageDraw

    n_f = min(len(gt_rgb), len(pr_rgb), len(cf_rgb))
    idxs = np.linspace(0, n_f - 1, n_cols).round().astype(int)
    h, w = 108, 180
    row_labels = ["GT", "PRED", "CF", "4x|P-CF|"]
    rows_img = []
    for r_i, lbl in enumerate(row_labels):
        tiles = []
        # Left label strip
        lbl_im = Image.new("RGB", (68, h), (16, 16, 20))
        ImageDraw.Draw(lbl_im).text((5, h // 2 - 6), lbl, fill=(240, 240, 245))
        tiles.append(np.asarray(lbl_im, dtype=np.uint8))
        for fi in idxs:
            if r_i == 0:
                arr = gt_rgb[fi]
            elif r_i == 1:
                arr = pr_rgb[fi]
            elif r_i == 2:
                arr = cf_rgb[fi]
            else:
                diff = np.abs(pr_rgb[fi].astype(np.float32) - cf_rgb[fi].astype(np.float32)) * 4.0
                arr = np.clip(diff, 0, 255).astype(np.uint8)
            im = Image.fromarray(arr).resize((w, h), Image.Resampling.BILINEAR)
            tiles.append(np.asarray(im, dtype=np.uint8))
        rows_img.append(np.concatenate(tiles, axis=1))
    body = np.concatenate(rows_img, axis=0)
    hdr = Image.new("RGB", (body.shape[1], 24), (10, 10, 14))
    draw = ImageDraw.Draw(hdr)
    draw.text((6, 5), title[:75], fill=(245, 245, 250))
    for ci, fi in enumerate(idxs):
        draw.text((68 + ci * w + 6, 5), f"t={fi / max(fps_assumed, 1.0):.2f}s (f{fi})", fill=(180, 220, 255))
    full = np.concatenate([np.asarray(hdr, dtype=np.uint8), body], axis=0)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(full).save(out_png)
    return str(out_png)


def _text_quality_metrics(text: str, *, is_code: bool = False, prompt: str = "") -> dict[str, Any]:
    toks = text.split()
    n_tok = len(toks)
    if n_tok >= 3:
        trigrams = [tuple(toks[i : i + 3]) for i in range(n_tok - 2)]
        distinct_3 = len(set(trigrams)) / float(len(trigrams))
    else:
        distinct_3 = 1.0 if n_tok > 0 else 0.0
    out: dict[str, Any] = {
        "n_chars": len(text),
        "n_words": n_tok,
        "distinct_3gram_ratio": round(distinct_3, 4),
    }
    if is_code:
        try:
            ast.parse(prompt + text)
            out["python_syntax_valid"] = True
        except SyntaxError:
            out["python_syntax_valid"] = False
    return out


# ---------------------------------------------------------------------------
# Main Long-Horizon Evaluation Orchestrator
# ---------------------------------------------------------------------------


def run_long_horizon_eval(shard_dir: Path, out_root: Path) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    models = [
        (
            "135m_routed",
            Path(".ops/checkpoints/confirmation_135m/confirmation_135m_slwm_v3_routed_c16_s4201_ckpt.pkl"),
            Path(".ops/checkpoints/confirmation_135m/wave_b_confirmation_135m_slwm_v3_routed_c16_s4201_ckpt.pkl"),
            Path(".ops/checkpoints/confirmation_135m/wave_c_confirmation_135m_slwm_v3_routed_c16_s4201_ckpt.pkl"),
        ),
        (
            "700m_routed",
            Path(".ops/checkpoints/scale_700m/scale_700m_slwm_v3_routed_c32_s4201_ckpt.pkl"),
            Path(".ops/checkpoints/scale_700m/wave_b_scale_700m_slwm_v3_routed_c32_s4201_ckpt.pkl"),
            Path(".ops/checkpoints/scale_700m/wave_c_scale_700m_slwm_v3_routed_c32_s4201_ckpt.pkl"),
        ),
        (
            "700m_single_bank",
            Path(".ops/checkpoints/scale_700m/scale_700m_single_bank_gated_deltanet_wy_s4201_ckpt.pkl"),
            Path(".ops/checkpoints/scale_700m/wave_b_scale_700m_single_bank_gated_deltanet_wy_s4201_ckpt.pkl"),
            Path(".ops/checkpoints/scale_700m/wave_c_scale_700m_single_bank_gated_deltanet_wy_s4201_ckpt.pkl"),
        ),
        (
            "700m_sw_gqa",
            Path(".ops/checkpoints/scale_700m/scale_700m_sliding_window_gqa_eqbyte_s4201_ckpt.pkl"),
            Path(".ops/checkpoints/scale_700m/wave_b_scale_700m_sliding_window_gqa_eqbyte_s4201_ckpt.pkl"),
            Path(".ops/checkpoints/scale_700m/wave_c_scale_700m_sliding_window_gqa_eqbyte_s4201_ckpt.pkl"),
        ),
        (
            "1b_routed",
            Path(".ops/checkpoints/scale_1b/scale_1b_slwm_v3_routed_c32_s4201_ckpt.pkl"),
            Path(".ops/checkpoints/scale_1b/wave_b_scale_1b_slwm_v3_routed_c32_s4201_ckpt.pkl"),
            Path(".ops/checkpoints/scale_1b/wave_c_scale_1b_slwm_v3_routed_c32_s4201_ckpt.pkl"),
        ),
    ]

    rob_sh = shard_dir / "robotics/robotics_test.npz"
    gh_sh = shard_dir / "av/greatesthits_test.npz"
    vgg_sh = shard_dir / "av/vggsound_test.npz"
    clotho_sh = shard_dir / "audio_text/clotho_test.npz"
    msrvtt_sh = shard_dir / "av/msrvtt_test.npz"

    report: dict[str, Any] = {
        "protocol": {
            "weights_strictly_frozen": True,
            "action2video_groups": 12,
            "action2video_frames": 45,
            "action2video_duration_s_25fps": 1.80,
            "audio2video_groups": 10,
            "audio2video_frames": 37,
            "audio2video_duration_s_12p5fps": 2.96,
            "video2audio_mimi_ticks_12p5hz": 50,
            "video2audio_duration_s": 4.00,
            "audio2audio_continuation_s": 4.00,
            "text2audio_duration_s": 4.00,
            "video2action_motor_steps": 60,
            "text_continuation_bytes": 256,
        },
        "models": {},
    }

    for tag, ck, wb, wc in models:
        print(f"\n================ LONG-HORIZON EVAL: {tag} ================", flush=True)
        mdir = out_root / tag
        mdir.mkdir(parents=True, exist_ok=True)
        bundle = load_frozen_bundle(ck, wave_b_ckpt=wb, wave_c_ckpt=wc)
        mres: dict[str, Any] = {}

        # 1. Long-horizon `action2video` (12 groups = 45 RGB frames = 1.80s @ 25fps / 3.0s @ 15fps!)
        act_res = stream_video_generation_from_shard(
            bundle, rob_sh, episode_idx=1, mode="action2video", n_pred_groups=12, out_dir=mdir / "action2video_1p8s"
        )
        z_a = np.load(mdir / "action2video_1p8s/action2video_ep0001_latents.npz")
        gt_rgb = decode_wan_patches_to_rgb(z_a["gt_patches"])
        pr_rgb = decode_wan_patches_to_rgb(z_a["pred_patches"])
        cf_rgb = decode_wan_patches_to_rgb(z_a["cf_patches"])
        assert gt_rgb is not None and pr_rgb is not None and cf_rgb is not None
        act_curve = _eval_video_rollout_curve(z_a["gt_patches"], z_a["pred_patches"], z_a["cf_patches"], gt_rgb, pr_rgb, cf_rgb, fps_assumed=25.0)
        act_strip = _save_temporal_filmstrip(
            gt_rgb, pr_rgb, cf_rgb, mdir / "action2video_1p8s/filmstrip_ep0001.png",
            title=f"{tag} action2video (45f / 1.80s @25fps)", fps_assumed=25.0,
        )
        # Re-save MP4 at smooth 15fps (3.0s playback) as well!
        _save_side_by_side_png_and_gif(
            [gt_rgb, pr_rgb, cf_rgb],
            mdir / "action2video_1p8s/action2video_ep0001_lastframe.png",
            mdir / "action2video_1p8s/action2video_ep0001_15fps_3p0s.gif",
            headers=["OBSERVED GT", f"{tag.upper()} PRED", "INVERTED ACTION CF"],
            fps=15.0,
        )
        mres["action2video_1p8s"] = {**act_res, "temporal_eval": act_curve, "filmstrip_png": act_strip}
        print(
            f"  [action2video 45f/1.80s] PSNR={act_curve['overall_psnr_db']} dB (vs CF {act_curve['overall_cf_psnr_db']} dB, "
            f"gain {act_curve['overall_psnr_gain_over_cf_db']:+.2f} dB) | early->late huber {act_curve['early_huber']}->{act_curve['late_huber']} "
            f"(drift {act_curve['drift_ratio_late_over_early']}x) | motion pred={act_curve['pred_frame_motion_rgb_l1']} vs gt={act_curve['gt_frame_motion_rgb_l1']}",
            flush=True,
        )

        # 2. Long-horizon `video2audio` 4.00s (50 Mimi ticks) on GreatestHits (physical impacts) AND VGGSound
        v2a_all = {}
        for ds_tag, sh_p, ep_i in [("greatesthits_ep0", gh_sh, 0), ("vggsound_ep0", vgg_sh, 0)]:
            vdir = mdir / f"video2audio_4s_{ds_tag}"
            vr = stream_cross_modal_from_shard(bundle, sh_p, episode_idx=ep_i, target_mode="audio", max_new_steps=50, out_dir=vdir)
            gt8_p = vdir / f"video2audio_ep{ep_i:04d}_gt_8rvq.wav"
            pr8_p = vdir / f"video2audio_ep{ep_i:04d}_pred_8rvq.wav"
            cf8_p = vdir / f"video2audio_ep{ep_i:04d}_shuffled_video_cf_8rvq.wav"
            codes_z = np.load(vdir / f"video2audio_ep{ep_i:04d}_mimi_codes.npz")
            dsp_gt = _audio_dsp_metrics(gt8_p)
            dsp_pr = _audio_dsp_metrics(pr8_p, ref_wav_path=gt8_p)
            dsp_cf = _audio_dsp_metrics(cf8_p, ref_wav_path=gt8_p)
            dsp_pr_vs_cf = _audio_dsp_metrics(pr8_p, ref_wav_path=cf8_p)
            div_pr = _rvq_code_diversity(codes_z["pred_codes"])
            div_gt = _rvq_code_diversity(codes_z["gt_codes_k"])
            spec_png = _save_spectrogram_panel(
                [(f"GT 8-RVQ ({ds_tag})", gt8_p), (f"{tag} PRED 4.0s", pr8_p), (f"{tag} SHUFFLED-VID CF 4.0s", cf8_p)],
                vdir / "spectrogram_4s_gt_vs_pred_vs_cf.png",
            )
            v2a_all[ds_tag] = {
                **vr,
                "dsp_gt": dsp_gt,
                "dsp_pred": dsp_pr,
                "dsp_cf": dsp_cf,
                "pred_vs_cf_log_mel_l1": dsp_pr_vs_cf.get("log_mel_l1_vs_ref"),
                "rvq_diversity_pred": div_pr,
                "rvq_diversity_gt": div_gt,
                "spectrogram_png": spec_png,
            }
            print(
                f"  [video2audio 4.0s {ds_tag}] melL1={dsp_pr.get('log_mel_l1_vs_ref')} (CF={dsp_cf.get('log_mel_l1_vs_ref')}), "
                f"env_r={dsp_pr.get('envelope_pearson_vs_ref')}, flat={dsp_pr.get('spectral_flatness')} (GT={dsp_gt.get('spectral_flatness')}), "
                f"RVQ0_ent={div_pr['rvq0_entropy_bits']}b (GT={div_gt['rvq0_entropy_bits']}b), rep={div_pr['rvq0_adj_repeat_rate']}",
                flush=True,
            )
        mres["video2audio_4s"] = v2a_all

        # 3. Long-horizon `audio2audio` (2.0s prefix -> 4.0s continuation = 6.0s total) on Clotho
        a2a_dir = mdir / "audio2audio_4s"
        a2a_res = stream_audio_continuation_from_shard(
            bundle, clotho_sh, episode_idx=0, prefix_ticks=25, gen_ticks=50, out_dir=a2a_dir
        )
        a2a_gt_p = a2a_dir / "audio2audio_ep0000_gt_8rvq.wav"
        a2a_pr_p = a2a_dir / "audio2audio_ep0000_pred_cont_8rvq.wav"
        a2a_z = np.load(a2a_dir / "audio2audio_ep0000_mimi_codes.npz")
        dsp_a2a_gt = _audio_dsp_metrics(a2a_gt_p)
        dsp_a2a_pr = _audio_dsp_metrics(a2a_pr_p, ref_wav_path=a2a_gt_p)
        div_a2a = _rvq_code_diversity(a2a_z["pred_codes"][25:])
        spec_a2a = _save_spectrogram_panel(
            [("GT Clotho 6.0s", a2a_gt_p), (f"{tag} 2s Prefix + 4s Cont", a2a_pr_p)],
            a2a_dir / "spectrogram_audio2audio_6s.png",
        )
        mres["audio2audio_4s"] = {
            **a2a_res,
            "dsp_gt": dsp_a2a_gt,
            "dsp_pred": dsp_a2a_pr,
            "rvq_diversity_suffix": div_a2a,
            "spectrogram_png": spec_a2a,
        }

        # 4. Long-horizon `text2audio` (4.00s = 50 ticks across 3 contrasting prompts -> prompt-sensitivity test!)
        t2a_dir = mdir / "text2audio_4s"
        t2a_prompts = [
            ("stream", "sound: water rushing in a stream"),
            ("dog", "sound: dog barking loudly"),
            ("drums", "sound: drum beats and percussion"),
        ]
        t2a_items = []
        spec_pairs = []
        for ptag, pr in t2a_prompts:
            tr = stream_text_to_audio(bundle, pr, max_new_steps=50, temperature=0.80, out_dir=t2a_dir, tag=ptag)
            wp = Path(tr["artifacts"]["pred_wav"])
            cz = np.load(t2a_dir / f"text2audio_{ptag}_mimi_codes.npz")
            dsp = _audio_dsp_metrics(wp)
            div = _rvq_code_diversity(cz["pred_codes"])
            t2a_items.append({**tr, "dsp": dsp, "rvq_diversity": div})
            spec_pairs.append((f"{tag} {ptag} 4.0s", wp))
        spec_t2a = _save_spectrogram_panel(spec_pairs, t2a_dir / "spectrogram_text2audio_3prompts_4s.png")
        centroids = [x["dsp"]["spectral_centroid_hz"] for x in t2a_items]
        mres["text2audio_4s"] = {
            "items": t2a_items,
            "prompt_centroid_spread_hz": round(max(centroids) - min(centroids), 1),
            "spectrogram_png": spec_t2a,
        }

        # 5. Long-horizon `audio2video` (10 groups = 37 RGB frames = 2.96s @ 12.5fps!) on VGGSound
        a2v_dir = mdir / "audio2video_3s"
        a2v_res = stream_video_generation_from_shard(
            bundle, vgg_sh, episode_idx=1, mode="audio2video", n_pred_groups=10, out_dir=a2v_dir
        )
        za2v = np.load(a2v_dir / "audio2video_ep0001_latents.npz")
        g_rgb = decode_wan_patches_to_rgb(za2v["gt_patches"])
        p_rgb = decode_wan_patches_to_rgb(za2v["pred_patches"])
        c_rgb = decode_wan_patches_to_rgb(za2v["cf_patches"])
        assert g_rgb is not None and p_rgb is not None and c_rgb is not None
        a2v_curve = _eval_video_rollout_curve(za2v["gt_patches"], za2v["pred_patches"], za2v["cf_patches"], g_rgb, p_rgb, c_rgb, fps_assumed=12.5)
        a2v_strip = _save_temporal_filmstrip(
            g_rgb, p_rgb, c_rgb, a2v_dir / "filmstrip_ep0001_3s.png",
            title=f"{tag} audio2video (37f / 2.96s @12.5fps)", fps_assumed=12.5,
        )
        mres["audio2video_3s"] = {**a2v_res, "temporal_eval": a2v_curve, "filmstrip_png": a2v_strip}

        # 6. Long-horizon `video2action` (60 continuous blind motor steps = 2.40s @ 25Hz!)
        v2act_res = stream_video_to_action_from_shard(bundle, rob_sh, episode_idx=1, max_steps=60)
        mres["video2action_2p4s"] = v2act_res

        # 7. Long-horizon `text2text`, `audio2text`, `video2text` (256 bytes!)
        pr_code = "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n"
        pr_prose = "The European Space Agency announced that the new telescope "
        t_code = stream_text_continuation(bundle, pr_code, max_new_bytes=256, temperature=0.0)
        t_prose = stream_text_continuation(bundle, pr_prose, max_new_bytes=256, temperature=0.0)
        a2t = stream_cross_modal_from_shard(bundle, clotho_sh, episode_idx=0, target_mode="text", max_new_steps=192)
        v2t = stream_cross_modal_from_shard(bundle, msrvtt_sh, episode_idx=0, target_mode="text", max_new_steps=192)
        mres["text_long_256b"] = {
            "code": {**t_code, "quality": _text_quality_metrics(t_code["generated"], is_code=True, prompt=pr_code)},
            "prose": {**t_prose, "quality": _text_quality_metrics(t_prose["generated"])},
            "audio2text": {**a2t, "quality": _text_quality_metrics(a2t["generated_text"])},
            "video2text": {**v2t, "quality": _text_quality_metrics(v2t["generated_text"])},
        }
        report["models"][tag] = mres

    # Synchronized 4-panel 1.80s (45-frame) 703M comparison MP4/GIF/filmstrip!
    comp_dir = out_root / "comparison_700m_1p8s"
    comp_dir.mkdir(parents=True, exist_ok=True)
    z_rt = np.load(out_root / "700m_routed/action2video_1p8s/action2video_ep0001_latents.npz")
    z_sb = np.load(out_root / "700m_single_bank/action2video_1p8s/action2video_ep0001_latents.npz")
    z_gq = np.load(out_root / "700m_sw_gqa/action2video_1p8s/action2video_ep0001_latents.npz")
    gt_r = decode_wan_patches_to_rgb(z_rt["gt_patches"])
    rt_r = decode_wan_patches_to_rgb(z_rt["pred_patches"])
    sb_r = decode_wan_patches_to_rgb(z_sb["pred_patches"])
    gq_r = decode_wan_patches_to_rgb(z_gq["pred_patches"])
    assert gt_r is not None and rt_r is not None and sb_r is not None and gq_r is not None
    _save_side_by_side_png_and_gif(
        [gt_r, rt_r, sb_r, gq_r],
        comp_dir / "action2video_45f_1p8s_gt_vs_routed_vs_sb_vs_gqa.png",
        comp_dir / "action2video_45f_1p8s_gt_vs_routed_vs_sb_vs_gqa.gif",
        headers=["OBSERVED GT (45f)", "703M SLWM-v3 ROUTED", "703M SINGLE-BANK", "703M SLIDING-WIN GQA"],
        fps=15.0,
    )

    out_json = out_root / "long_horizon_eval_report.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nLONG_HORIZON_EVAL_OK -> {out_json}", flush=True)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", type=Path, default=Path(".ops/v2_shards"))
    ap.add_argument("--out-dir", type=Path, default=Path(".ops/m4max_v2_long_horizon_eval"))
    args = ap.parse_args()
    run_long_horizon_eval(args.shard_dir, args.out_dir)
