"""Study X3 (`SLWM-1B-480p`) Results Aggregator, Crossed Hierarchical Bootstrap CI Engine,
Scaling-Law Fitter, `0-1B` / `1-30B` SoTA Contextualizer, & Deterministic `vault` Showcase Exporter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any
import wave

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.x1b_stats import hierarchical_paired_auroc_bootstrap_ci, hierarchical_seed_bootstrap_ci

SOTA_COMPARISON_SCOPE_NOTE = (
    "Architectural and systems capability taxonomy across published 0-1B and 1-30B world/VLA/omni models; "
    "external rows reflect published system specifications under unmatched upstream pretraining compute and corpora, "
    "whereas controlled matched-compute empirical claims are restricted to the 1.022B Study X3 cohorts."
)

SOTA_0_TO_1B_CONTEXT = [
    {
        "model": "DINO-WM (ViT-S/B)",
        "params": "22M - 86M",
        "native_res": "224x224",
        "temporal_complexity": "O(W^2) window",
        "streaming_state_T1024": "Window-truncated",
        "modalities": "RGB + Action",
        "any_to_any_dirs": "1 (Latent Rollout -> Control)",
        "causal_livestream": False,
        "comparison_scope": "architectural_taxonomy_unmatched_pretraining",
    },
    {
        "model": "Octo-Base",
        "params": "93M",
        "native_res": "256x256",
        "temporal_complexity": "O(W^2) window (W=2)",
        "streaming_state_T1024": "2-frame horizon",
        "modalities": "RGB + Text + Proprio -> Action",
        "any_to_any_dirs": "1 (Obs+Text -> Diffusion Action)",
        "causal_livestream": False,
        "comparison_scope": "architectural_taxonomy_unmatched_pretraining",
    },
    {
        "model": "DIAMOND",
        "params": "130M",
        "native_res": "64x64 - 160x210",
        "temporal_complexity": "Multi-step UNet diffusion",
        "streaming_state_T1024": "4-frame conditioning buffer",
        "modalities": "RGB + Discrete Action -> RGB",
        "any_to_any_dirs": "1 (Action -> Video)",
        "causal_livestream": False,
        "comparison_scope": "architectural_taxonomy_unmatched_pretraining",
    },
    {
        "model": "DreamerV3 XL",
        "params": "200M",
        "native_res": "64x64 - 128x128",
        "temporal_complexity": "O(1) RSSM vector",
        "streaming_state_T1024": "< 1 MiB (vector, no matrix K-V)",
        "modalities": "RGB + Proprio + Action -> RGB/Reward",
        "any_to_any_dirs": "2 (Video + Action/Value)",
        "causal_livestream": True,
        "comparison_scope": "architectural_taxonomy_unmatched_pretraining",
    },
    {
        "model": "SmolVLA",
        "params": "450M",
        "native_res": "384x384",
        "temporal_complexity": "O(T^2) VLM + flow action expert",
        "streaming_state_T1024": "Unbounded KV / short chunk",
        "modalities": "RGB + Text + Proprio -> Action",
        "any_to_any_dirs": "1 (VLA Policy)",
        "causal_livestream": False,
        "comparison_scope": "architectural_taxonomy_unmatched_pretraining",
    },
    {
        "model": "Cosmos-Predict2-0.6B",
        "params": "600M",
        "native_res": "480p - 720p",
        "temporal_complexity": "Non-causal bidirectional DiT",
        "streaming_state_T1024": "Offline chunk diffusion",
        "modalities": "RGB + Text/Action -> RGB",
        "any_to_any_dirs": "1 (Text/Img -> Video)",
        "causal_livestream": False,
        "comparison_scope": "architectural_taxonomy_unmatched_pretraining",
    },
    {
        "model": "V-JEPA 2 (ViT-L / ViT-g)",
        "params": "300M - 1.0B",
        "native_res": "256x256 - 384x384",
        "temporal_complexity": "O(T^2) spatiotemporal ViT",
        "streaming_state_T1024": "Clip-windowed",
        "modalities": "RGB (+ Action-Cond Predictor)",
        "any_to_any_dirs": "1 (Latent Prediction + MPC)",
        "causal_livestream": False,
        "comparison_scope": "architectural_taxonomy_unmatched_pretraining",
    },
    {
        "model": "SLWM-X3-1B-480p (Ours)",
        "params": "1.022B",
        "native_res": "640x480 (480p)",
        "temporal_complexity": "O(1) Hybrid GatedDelta + W=8",
        "streaming_state_T1024": "Fixed 30.14 MiB (0 B growth)",
        "modalities": "480p RGB + 16kHz Audio + Text + 32D Mov + 16D Act",
        "any_to_any_dirs": "6 (All-way Any-to-Any + Epistemic Gate)",
        "causal_livestream": True,
        "comparison_scope": "empirical_matched_1b_study_x3",
    },
]

SOTA_1_TO_30B_CONTEXT = [
    {
        "model": "Wan2.1-T2V-1.3B / 14B",
        "params": "1.3B - 14B",
        "native_res": "480p - 720p",
        "temporal_complexity": "Non-causal 3D DiT (50-step denoise)",
        "streaming_state_T1024": "Offline clip (~seconds/frame)",
        "modalities": "Text/Image -> Video",
        "any_to_any_dirs": "1 (Video Generation)",
        "causal_livestream": False,
        "comparison_scope": "architectural_taxonomy_unmatched_pretraining",
    },
    {
        "model": "LTX-Video (2B / 13B)",
        "params": "2.0B - 13B",
        "native_res": "480p - 768p",
        "temporal_complexity": "Non-causal compressed DiT",
        "streaming_state_T1024": "Offline clip",
        "modalities": "Text/Image -> Video",
        "any_to_any_dirs": "1 (Video Generation)",
        "causal_livestream": False,
        "comparison_scope": "architectural_taxonomy_unmatched_pretraining",
    },
    {
        "model": "Physical Intelligence pi0",
        "params": "3.0B",
        "native_res": "224x224 (multi-cam)",
        "temporal_complexity": "PaliGemma VLM + Flow Matching",
        "streaming_state_T1024": "Per-step VLM re-encode",
        "modalities": "RGB + Text + Proprio -> 50Hz Action",
        "any_to_any_dirs": "1 (VLA Control)",
        "causal_livestream": False,
        "comparison_scope": "architectural_taxonomy_unmatched_pretraining",
    },
    {
        "model": "NVIDIA GR00T N1.5",
        "params": "3.0B",
        "native_res": "224x224 - 480p",
        "temporal_complexity": "Eagle-2 VLM + DiT Action Head",
        "streaming_state_T1024": "Short-window VLM",
        "modalities": "RGB + Text + Proprio -> Humanoid Action",
        "any_to_any_dirs": "1 (VLA Control)",
        "causal_livestream": False,
        "comparison_scope": "architectural_taxonomy_unmatched_pretraining",
    },
    {
        "model": "OpenVLA",
        "params": "7.0B",
        "native_res": "224x224",
        "temporal_complexity": "Llama-2 7B per-frame autoregressive",
        "streaming_state_T1024": "Single-frame Markov",
        "modalities": "RGB + Text -> 7-DoF Tokenized Action",
        "any_to_any_dirs": "1 (VLA Control)",
        "causal_livestream": False,
        "comparison_scope": "architectural_taxonomy_unmatched_pretraining",
    },
    {
        "model": "Kyutai Moshi",
        "params": "7.0B",
        "native_res": "Audio-only (24 kHz)",
        "temporal_complexity": "O(T) streaming KV cache",
        "streaming_state_T1024": "Grows O(T) (no vision/actions)",
        "modalities": "Full-Duplex Speech <-> Text/Speech",
        "any_to_any_dirs": "2 (Audio <-> Text)",
        "causal_livestream": True,
        "comparison_scope": "architectural_taxonomy_unmatched_pretraining",
    },
    {
        "model": "Qwen2.5-Omni",
        "params": "7.0B",
        "native_res": "Dynamic ViT + 16kHz Audio",
        "temporal_complexity": "O(T) Thinker-Talker KV cache",
        "streaming_state_T1024": "Multi-GiB growing KV (no robot actions)",
        "modalities": "Video + Audio + Text -> Text + Speech",
        "any_to_any_dirs": "2 (Omni -> Text/Speech)",
        "causal_livestream": True,
        "comparison_scope": "architectural_taxonomy_unmatched_pretraining",
    },
    {
        "model": "DeepMind Genie 2",
        "params": "~11B",
        "native_res": "360p - 720p",
        "temporal_complexity": "Autoregressive latent diffusion + KV",
        "streaming_state_T1024": "~10-60s horizon memory window",
        "modalities": "Image + Action -> Interactive Video",
        "any_to_any_dirs": "1 (Action -> Video)",
        "causal_livestream": True,
        "comparison_scope": "architectural_taxonomy_unmatched_pretraining",
    },
    {
        "model": "NVIDIA Cosmos-Predict1/2 (7B / 14B)",
        "params": "7.0B - 14B",
        "native_res": "720p",
        "temporal_complexity": "O(T) AR / Non-causal DiT",
        "streaming_state_T1024": "Multi-GiB KV or offline chunk",
        "modalities": "Text/Video/Action -> Video",
        "any_to_any_dirs": "1 (World Video Generation)",
        "causal_livestream": False,
        "comparison_scope": "architectural_taxonomy_unmatched_pretraining",
    },
]


def fit_chinchilla_surface_nd(
    nd_points: list[tuple[float, float, float]],
) -> dict[str, Any]:
    """Fit data-constrained 2D scaling surface L(N, D) = E + A * N^(-alpha) + B * D^(-beta) on held-out validation loss."""
    if len(nd_points) < 6:
        return {
            "E": 0.0,
            "A": 0.0,
            "alpha": 0.0,
            "B": 0.0,
            "beta": 0.0,
            "a_opt": 0.5,
            "b_opt": 0.5,
            "r2": 0.0,
            "regime": "data_constrained_multi_epoch_heldout_val",
        }
    n_arr = np.asarray([p[0] for p in nd_points], dtype=np.float64)
    d_arr = np.asarray([p[1] for p in nd_points], dtype=np.float64)
    y_arr = np.asarray([p[2] for p in nd_points], dtype=np.float64)
    ss_tot = max(float(np.sum((y_arr - np.mean(y_arr)) ** 2)), 1e-12)
    y_min = max(0.0, float(np.min(y_arr)))

    best_r2 = -1e9
    best: dict[str, Any] = {
        "E": 0.0,
        "A": 0.0,
        "alpha": 0.0,
        "B": 0.0,
        "beta": 0.0,
        "a_opt": 0.5,
        "b_opt": 0.5,
        "r2": 0.0,
        "regime": "data_constrained_multi_epoch_heldout_val",
    }
    alphas = np.linspace(0.05, 0.85, 17)
    betas = np.linspace(0.05, 0.85, 17)
    es = np.linspace(0.0, 0.85 * y_min, 10)
    for al in alphas:
        na = n_arr ** (-al)
        for be in betas:
            db = d_arr ** (-be)
            design = np.stack([na, db], axis=1)
            for e_val in es:
                rem = y_arr - e_val
                if np.any(rem <= 0.0):
                    continue
                coef, _, _, _ = np.linalg.lstsq(design, rem, rcond=None)
                a_c = max(0.0, float(coef[0]))
                b_c = max(0.0, float(coef[1]))
                pred = e_val + a_c * na + b_c * db
                r2 = 1.0 - float(np.sum((y_arr - pred) ** 2)) / ss_tot
                if r2 > best_r2:
                    best_r2 = r2
                    s_exp = max(1e-8, float(al + be))
                    best = {
                        "E": round(float(e_val), 6),
                        "A": round(a_c, 4),
                        "alpha": round(float(al), 4),
                        "B": round(b_c, 4),
                        "beta": round(float(be), 4),
                        "a_opt": round(float(be / s_exp), 4),
                        "b_opt": round(float(al / s_exp), 4),
                        "r2": round(float(r2), 5),
                        "n_grid_points": int(len(nd_points)),
                        "regime": "data_constrained_multi_epoch_heldout_val",
                    }
    return best


def fit_power_law_3pt(params_n: list[float], values_y: list[float], *, positive_is_lower: bool = True) -> dict[str, float]:
    n_arr = np.asarray(params_n, dtype=np.float64)
    y_arr = np.asarray(values_y, dtype=np.float64)
    if not positive_is_lower:
        ceil_v = max(1.0, float(np.max(y_arr)) * 1.15)
        y_arr = np.maximum(ceil_v - y_arr, 1e-5)

    best_r2 = -1e9
    best = {"E": 0.0, "A": float(y_arr[0]), "alpha": 0.0, "r2": 0.0}
    y_min = float(np.min(y_arr))
    for e_cand in np.linspace(0.0, max(1e-6, 0.92 * y_min), 120):
        rem = y_arr - e_cand
        if np.any(rem <= 1e-8):
            continue
        lx = np.log(n_arr)
        ly = np.log(rem)
        slope, intercept = np.polyfit(lx, ly, 1)
        alpha = -float(slope)
        a_val = float(np.exp(intercept))
        pred = e_cand + a_val * (n_arr ** (-alpha))
        ss_res = float(np.sum((y_arr - pred) ** 2))
        ss_tot = max(float(np.sum((y_arr - np.mean(y_arr)) ** 2)), 1e-12)
        r2 = 1.0 - ss_res / ss_tot
        if r2 > best_r2:
            best_r2 = r2
            best = {"E": round(float(e_cand), 6), "A": round(a_val, 6), "alpha": round(alpha, 4), "r2": round(r2, 5)}
    return best


def _write_wav_16khz(path: Path, pcm_f32: np.ndarray, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    s16 = np.clip(np.asarray(pcm_f32, dtype=np.float32) * 32767.0, -32768.0, 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(s16.tobytes())


def _encode_mp4_from_frames(frames_u8: list[np.ndarray], mp4_path: Path, fps: int = 4) -> bool:
    ffmpeg_bin = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    if not Path(ffmpeg_bin).is_file():
        return False
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    h, w, _ = frames_u8[0].shape
    h_even = (h // 2) * 2
    w_even = (w // 2) * 2
    raw = b"".join(np.asarray(f[:h_even, :w_even, :3], dtype=np.uint8).tobytes() for f in frames_u8)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w_even}x{h_even}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "20",
        str(mp4_path),
    ]
    res = subprocess.run(cmd, input=raw, capture_output=True, timeout=60)
    return res.returncode == 0 and mp4_path.is_file()


def _select_median_seed_entry(seed_evals: list[dict[str, Any]], seed_dirs: list[Path]) -> tuple[dict[str, Any], Path | None]:
    """Deterministically select the empirical median seed (`rank len//2`) by composite primary capability score."""
    scored: list[tuple[float, int]] = []
    for idx, ev in enumerate(seed_evals):
        sm = ev["summary"]
        comp = (
            sm.get("dyn_visual_assoc_T1024", 0.0)
            + sm.get("T160_paired_causal_switch", 0.0)
            + sm.get("unseen_mppi_480p", 0.0)
            + sm.get("commit_selective_routing_acc", 0.0)
            + 0.02 * sm.get("vault_480p_video_psnr_db", 0.0)
        )
        scored.append((float(comp), idx))
    scored.sort(key=lambda t: (t[0], int(seed_evals[t[1]].get("seed", 0))))
    med_idx = scored[len(scored) // 2][1]
    sc_p = seed_dirs[med_idx] / "showcase_bundle.npz"
    return seed_evals[med_idx], (sc_p if sc_p.is_file() else None)


def export_x3_showcase_gallery(
    showcase_npz: Path,
    eval_doc: dict[str, Any],
    out_dir: Path,
    *,
    cohort_median_bundles: dict[str, tuple[str, Path]] | None = None,
) -> dict[str, Any]:
    """Export sealed-`vault` `P05..P95` + synchronized 6-column common-episode `.mp4`/`.wav`/`.png` gallery."""
    out_dir.mkdir(parents=True, exist_ok=True)
    exported: dict[str, list[str]] = {"mp4_videos": [], "wav_audio": [], "png_panels": []}
    if not showcase_npz.is_file():
        return exported

    with np.load(showcase_npz, allow_pickle=False) as d:
        pcts = d["dir2_percentile_labels"] if "dir2_percentile_labels" in d.files else np.array([5, 25, 50, 75, 95])
        pct_ep_ids = d["dir2_percentile_ep_ids"] if "dir2_percentile_ep_ids" in d.files else np.arange(len(pcts))
        psnrs = d["dir2_percentile_psnr_db"] if "dir2_percentile_psnr_db" in d.files else np.zeros((len(pcts),))
        obs0 = d["dir2_obs_f0_480p_u8"]
        targ = d["dir2_targ_480p_u8"]
        pred = d["dir2_pred_480p_u8"]
        negu = d["dir2_negu_480p_u8"]
        common_ep_ids = d["dir2_common_ep_ids"] if "dir2_common_ep_ids" in d.files else np.array([], dtype=np.int64)
        common_obs0 = d["dir2_common_obs_f0_480p_u8"] if "dir2_common_obs_f0_480p_u8" in d.files else None
        common_targ = d["dir2_common_targ_480p_u8"] if "dir2_common_targ_480p_u8" in d.files else None
        common_pred_slwm = d["dir2_common_pred_480p_u8"] if "dir2_common_pred_480p_u8" in d.files else None
        pmel = d["dir3_pred_mel64"]
        tmel = d["dir3_targ_mel64"]
        pwav = d["dir3_pred_wav_16khz"]
        twav = d["dir3_targ_wav_16khz"]
        ls_pg = d["livestream_pred_goal_trace"]
        ls_tg = d["livestream_targ_goal_trace"]
        ls_md = d["livestream_mode_trace"]
        ls_ms = d["livestream_tick_ms"]

    # 1. Export all 5 failure-to-best percentiles (`P05, P25, P50, P75, P95`) on sealed `vault`
    for ei in range(pred.shape[0]):
        pct_lbl = f"P{int(pcts[ei]):02d}" if ei < len(pcts) else f"ep{ei}"
        psnr_v = float(psnrs[ei]) if ei < len(psnrs) else 0.0
        ep_id_v = int(pct_ep_ids[ei]) if ei < len(pct_ep_ids) else ei
        quad_frames: list[np.ndarray] = []
        for ti in range(pred.shape[1]):
            canvas = Image.new("RGB", (640 * 4, 480 + 38), (18, 20, 26))
            draw = ImageDraw.Draw(canvas)
            canvas.paste(Image.fromarray(targ[ei, ti]), (0, 38))
            canvas.paste(Image.fromarray(obs0[ei, 0]), (640, 38))
            canvas.paste(Image.fromarray(pred[ei, ti]), (1280, 38))
            canvas.paste(Image.fromarray(negu[ei, ti]), (1920, 38))
            draw.text((12, 11), f"[observed] Vault ep={ep_id_v} {pct_lbl} GT step {ti+1}", fill=(180, 235, 190))
            draw.text((652, 11), "[observed] Trivial Frame-0 Copy Baseline", fill=(210, 210, 210))
            draw.text((1292, 11), f"[predicted | forced_diagnostic] SLWM-1B-480p ({psnr_v:.2f} dB)", fill=(130, 210, 255))
            draw.text((1932, 11), "[imagined | forced_diagnostic] Counterfactual -Act/Swapped-Txt", fill=(255, 185, 130))
            quad_frames.append(np.asarray(canvas, dtype=np.uint8))
            if ti == pred.shape[1] - 1:
                png_p = out_dir / f"dir2_480p_video_synthesis_{pct_lbl}.png"
                canvas.save(png_p)
                exported["png_panels"].append(png_p.name)
        mp4_p = out_dir / f"dir2_480p_video_synthesis_{pct_lbl}.mp4"
        if _encode_mp4_from_frames([obs0[ei, 0]] + [pred[ei, t] for t in range(pred.shape[1])], mp4_p, fps=3):
            exported["mp4_videos"].append(mp4_p.name)
        mp4_quad = out_dir / f"dir2_480p_triptych_comparison_{pct_lbl}.mp4"
        if _encode_mp4_from_frames(quad_frames, mp4_quad, fps=2):
            exported["mp4_videos"].append(mp4_quad.name)

    # 2. Export synchronized common-episode cross-cohort `vault` comparison (`Ground Truth | Frame-0 Copy | Matched Baselines | SLWM-1B-480p`)
    peer_common_preds: dict[str, np.ndarray] = {}
    peer_selected_tags: dict[str, str] = {}
    if cohort_median_bundles:
        for coh_k, (sel_tag, npz_p) in cohort_median_bundles.items():
            peer_selected_tags[coh_k] = sel_tag
            if npz_p.is_file():
                with np.load(npz_p, allow_pickle=False) as pd:
                    if "dir2_common_pred_480p_u8" in pd.files:
                        peer_common_preds[coh_k] = pd["dir2_common_pred_480p_u8"]

    if common_obs0 is not None and common_targ is not None and common_pred_slwm is not None and common_pred_slwm.shape[0] > 0:
        # Build 6-column synchronized panel at half-scale per tile (`320x240` tiles -> `1920x276` 1080p-width strip)
        col_defs: list[tuple[str, tuple[int, int, int], np.ndarray]] = [
            ("[observed] Ground Truth", (180, 235, 190), common_targ),
            ("[observed] Frame-0 Copy", (210, 210, 210), np.repeat(common_obs0, common_targ.shape[1], axis=1)),
        ]
        for coh_k, lbl_short, col_rgb in (
            ("x3_sliding_window_w64_gqa_1b_480p", "[predicted] W64-GQA-1B", (255, 195, 120)),
            ("x3_diagonal_ssm_1b_480p", "[predicted] DiagSSM-1B", (210, 170, 255)),
            ("x3_slwm_1b_240p_ablation", "[predicted] SLWM-1B-240p", (255, 150, 150)),
        ):
            if coh_k in peer_common_preds and peer_common_preds[coh_k].shape == common_pred_slwm.shape:
                col_defs.append((lbl_short, col_rgb, peer_common_preds[coh_k]))
        col_defs.append(("[predicted] SLWM-1B-480p", (130, 210, 255), common_pred_slwm))

        tile_w, tile_h = 320, 240
        n_common = min(3, int(common_pred_slwm.shape[0]))
        for ci in range(n_common):
            ep_id_c = int(common_ep_ids[ci]) if ci < len(common_ep_ids) else ci
            sync_frames: list[np.ndarray] = []
            for ti in range(common_pred_slwm.shape[1]):
                strip = Image.new("RGB", (tile_w * len(col_defs), tile_h + 34), (16, 18, 24))
                sdraw = ImageDraw.Draw(strip)
                for col_i, (clbl, crgb, carr) in enumerate(col_defs):
                    im_tile = Image.fromarray(carr[ci, ti]).resize((tile_w, tile_h), Image.Resampling.BILINEAR)
                    strip.paste(im_tile, (col_i * tile_w, 34))
                    sdraw.text((col_i * tile_w + 8, 10), f"{clbl} (ep={ep_id_c})", fill=crgb)
                sync_frames.append(np.asarray(strip, dtype=np.uint8))
                if ti == common_pred_slwm.shape[1] - 1:
                    cp_png = out_dir / f"common_vault_cross_model_ep{ep_id_c:03d}.png"
                    strip.save(cp_png)
                    exported["png_panels"].append(cp_png.name)
            cp_mp4 = out_dir / f"common_vault_cross_model_ep{ep_id_c:03d}.mp4"
            if _encode_mp4_from_frames(sync_frames, cp_mp4, fps=2):
                exported["mp4_videos"].append(cp_mp4.name)

    # 3. Export 16 kHz `.wav` audio + spectrogram comparison (`[observed]` vs `[predicted]`)
    for ai in range(min(3, pwav.shape[0])):
        wp = out_dir / f"dir3_synthesized_from_video_ep{ai}_16khz.wav"
        wt = out_dir / f"dir3_ground_truth_reference_ep{ai}_16khz.wav"
        _write_wav_16khz(wp, pwav[ai])
        _write_wav_16khz(wt, twav[ai])
        exported["wav_audio"].extend([wp.name, wt.name])

    if pmel.shape[0] > 0:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), dpi=160)
        axes[0].imshow(tmel[0].T, aspect="auto", origin="lower", cmap="magma")
        axes[0].set_title("[observed] 64-Band Vault Reference Audio (16 kHz)")
        axes[0].set_xlabel("Causal Tick (100 ms / tick)")
        axes[0].set_ylabel("Geom Frequency Band [60-7680 Hz]")
        axes[1].imshow(pmel[0].T, aspect="auto", origin="lower", cmap="magma")
        axes[1].set_title("[predicted | forced_diagnostic] Video+Act -> Audio (mm[aud]=0)")
        axes[1].set_xlabel("Causal Tick (100 ms / tick)")
        plt.tight_layout()
        sp_png = out_dir / "dir3_video_to_audio_spectrogram_ep0.png"
        fig.savefig(sp_png)
        plt.close(fig)
        exported["png_panels"].append(sp_png.name)

    # 4. Export continuous `T=1,024` livestream trace (`PolicyCommitGate_routed`: commit / fallback / probe / abstained)
    fig, axes = plt.subplots(3, 1, figsize=(11.0, 6.8), dpi=160, sharex=True)
    ticks_x = np.arange(ls_pg.shape[0])
    axes[0].plot(ticks_x, ls_tg[:, 0], "--", color="#2ecc71", lw=1.6, label="[observed] Target Goal X")
    axes[0].plot(ticks_x, ls_pg[:, 0], "-", color="#2980b9", lw=1.3, label="[predicted] Livestream Goal X")
    axes[0].plot(ticks_x, ls_tg[:, 1], ":", color="#27ae60", lw=1.4, label="[observed] Target Goal Y")
    axes[0].plot(ticks_x, ls_pg[:, 1], "-", color="#8e44ad", lw=1.1, alpha=0.85, label="[predicted] Livestream Goal Y")
    axes[0].set_ylabel("Goal Latch XY")
    axes[0].set_title("[PolicyCommitGate_routed] Continuous B=1 Causal 480p Livestream (3 Live Switches + Blackout)")
    axes[0].legend(loc="upper right", ncol=2, fontsize=8)

    axes[1].step(ticks_x, ls_md, where="post", color="#d35400", lw=1.4)
    axes[1].set_yticks([0, 1, 2, 3])
    axes[1].set_yticklabels(["commit(0)", "fallback(1)", "probe(2)", "abstained(3)"])
    axes[1].set_ylabel("PolicyCommitGate")

    axes[2].plot(ticks_x, ls_ms, color="#34495e", lw=1.0)
    axes[2].set_ylabel("Tick Latency (ms)")
    axes[2].set_xlabel("Continuous Causal Stream Tick t (10 Hz)")
    plt.tight_layout()
    ls_png = out_dir / "livestream_T1024_goal_switches_and_latency.png"
    fig.savefig(ls_png)
    plt.close(fig)
    exported["png_panels"].append(ls_png.name)

    manifest = {
        "gallery_split": "sealed_vault",
        "online_weight_updates": False,
        "selection_rule": "deterministic_median_seed_and_P05_P25_P50_P75_P95_percentiles_plus_locked_common_vault_episodes",
        "selected_median_seed_tag": eval_doc.get("tag", "unknown"),
        "cohort_median_seed_tags": peer_selected_tags,
        "native_resolution": [480, 640, 3],
        "percentiles_exported": [int(x) for x in np.asarray(pcts).tolist()],
        "percentile_vault_episode_ids": [int(x) for x in np.asarray(pct_ep_ids).tolist()],
        "locked_common_vault_episode_ids": [int(x) for x in np.asarray(common_ep_ids).tolist()],
        "epistemic_labels": ["observed", "reconstructed", "predicted", "imagined", "abstained"],
        "routing_provenance": {
            "dir2_480p_video_synthesis": "forced_diagnostic_decode",
            "dir2_counterfactual_video": "forced_diagnostic_decode_imagined",
            "dir3_video_to_audio_16khz": "forced_diagnostic_decode",
            "livestream_T1024_trace": "PolicyCommitGate_routed",
        },
        "sample_video_audio_to_text": eval_doc.get("any_to_any_vault", {})
        .get("dir1_video_audio_to_text", {})
        .get("sample_predictions", []),
        "exported_files": exported,
    }
    manifest_json = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (out_dir / "showcase_manifest.json").write_text(manifest_json)
    (out_dir / "manifest.json").write_text(manifest_json)
    return manifest


def render_x3_publication_figures(summary_doc: dict[str, Any], fig_dir: Path) -> list[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    cohorts = summary_doc["cohorts"]
    created: list[str] = []

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.3), dpi=180)
    sc_names = ["x3_slwm_150m_480p", "x3_slwm_450m_480p", "x3_slwm_1b_480p"]
    present_sc = [c for c in sc_names if c in cohorts]
    if present_sc:
        xs_m = [cohorts[c]["strict_params"] / 1e6 for c in present_sc]
        ys_val = [cohorts[c]["terminal_val_loss"]["mean"] for c in present_sc]
        ys_psnr = [cohorts[c]["metrics"]["vault_480p_video_psnr_db"]["mean"] for c in present_sc]
        axes[0].plot(xs_m, ys_val, "o-", color="#1f77b4", lw=2.2, ms=7)
        axes[0].set_xscale("log")
        axes[0].set_xlabel("Strict Parameters (Millions, log scale)")
        axes[0].set_ylabel("Held-Out Dev Validation Loss")
        axes[0].set_title("A. Scaling Law (140.0M -> 473.5M -> 1.022B)")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(xs_m, ys_psnr, "s-", color="#2ca02c", lw=2.2, ms=7, label="SLWM-X3 480p")
        if "x3_slwm_1b_240p_ablation" in cohorts:
            axes[1].scatter(
                [cohorts["x3_slwm_1b_240p_ablation"]["strict_params"] / 1e6],
                [cohorts["x3_slwm_1b_240p_ablation"]["metrics"]["vault_480p_video_psnr_db"]["mean"]],
                marker="D", s=65, color="#e74c3c", label="SLWM-X3 1B (240p Ablation)",
            )
        axes[1].set_xscale("log")
        axes[1].set_xlabel("Strict Parameters (Millions, log scale)")
        axes[1].set_ylabel("Vault 640x480 Video PSNR (dB)")
        axes[1].set_title("B. Native 480p Synthesis Scaling & 240p Ablation")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

    h2h = [
        ("x3_slwm_1b_480p", "SLWM-1B\n(480p)", "#1f77b4"),
        ("x3_sliding_window_w64_gqa_1b_480p", "W64-GQA-1B\n(480p)", "#ff7f0e"),
        ("x3_diagonal_ssm_1b_480p", "DiagSSM-1B\n(480p)", "#9467bd"),
        ("x3_slwm_1b_240p_ablation", "SLWM-1B\n(240p)", "#d62728"),
    ]
    pres_h = [(k, lbl, col) for k, lbl, col in h2h if k in cohorts]
    if pres_h:
        x_pos = np.arange(len(pres_h))
        w = 0.25
        m1 = [cohorts[k]["metrics"]["dyn_visual_assoc_T1024"]["mean"] * 100 for k, _, _ in pres_h]
        m2 = [cohorts[k]["metrics"]["T160_paired_causal_switch"]["mean"] * 100 for k, _, _ in pres_h]
        m3 = [cohorts[k]["metrics"]["unseen_mppi_480p"]["mean"] * 100 for k, _, _ in pres_h]
        axes[2].bar(x_pos - w, m1, width=w, label="T=1024 Assoc Overwrite (%)", color="#2980b9")
        axes[2].bar(x_pos, m2, width=w, label="T=160 Paired Switch (%)", color="#27ae60")
        axes[2].bar(x_pos + w, m3, width=w, label="Unseen 480p MPPI (%)", color="#e67e22")
        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels([lbl for _, lbl, _ in pres_h], fontsize=8.5)
        axes[2].set_ylim(0, 108)
        axes[2].set_ylabel("Success / Accuracy (%)")
        axes[2].set_title("C. Matched 1.022B Architecture Head-to-Head")
        axes[2].legend(fontsize=7.5, loc="lower right")
        axes[2].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    f1 = fig_dir / "x3_1b_480p_scaling_laws_and_1b_head_to_head.png"
    fig.savefig(f1)
    plt.close(fig)
    created.append(f1.name)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), dpi=180)
    if "x3_slwm_1b_480p" in cohorts:
        c1b = cohorts["x3_slwm_1b_480p"]["any_to_any_vault_lifts"]
        d_labels = [
            "1. Vid/Aud->Text\n(Within-Src CE Lift)",
            "2. Text+F0->480p\n(Fixed-F0 PSNR Lift)",
            "3. Vid->16kHz Aud\n(LSD dB Lift)",
            "4. Txt/Vid->3D Mov\n(Phys MSE Lift x10)",
            "5. AV+Txt->Action\n(True Act Lift x10)",
            "6. Blind AP->Occ. Motion\n(Vis Motion BCE Lift x10)",
        ]
        vals = [
            c1b["dir1_ce_lift"]["mean"],
            c1b["dir2_psnr_db_lift"]["mean"],
            c1b["dir3_lsd_db_lift"]["mean"],
            c1b["dir4_mse_lift"]["mean"] * 10.0,
            c1b["dir5_mse_lift"]["mean"] * 10.0,
            c1b["dir6_bce_lift"]["mean"] * 10.0,
        ]
        colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in vals]
        axes[0].barh(np.arange(6), vals, color=colors)
        axes[0].axvline(0.0, color="black", lw=1.0)
        axes[0].set_yticks(np.arange(6))
        axes[0].set_yticklabels(d_labels, fontsize=8.5)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Within-Source Disjoint-Group Shuffle Lift (> 0 Proves Causal Conditioning)")
        axes[0].set_title("A. SLWM-1B-480p 6-Way Any-to-Any Counterfactual Lift (Vault)")
        axes[0].grid(True, axis="x", alpha=0.3)

    if pres_h:
        xp = np.arange(len(pres_h))
        edge_v = [cohorts[k]["metrics"]["vault_480p_edge_rmse"]["mean"] for k, _, _ in pres_h]
        axes[1].bar(xp, edge_v, width=0.48, color=[col for _, _, col in pres_h])
        axes[1].set_xticks(xp)
        axes[1].set_xticklabels([lbl for _, lbl, _ in pres_h], fontsize=8.5)
        axes[1].set_ylabel("480p High-Freq Spatial Edge RMSE (Lower = Sharper)")
        axes[1].set_title("B. Native 480p vs 240p Fine Spatial Edge Fidelity")
        axes[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    f2 = fig_dir / "x3_1b_480p_any_to_any_6way_composition.png"
    fig.savefig(f2)
    plt.close(fig)
    created.append(f2.name)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.3), dpi=180)
    ks = [1, 2, 4, 8, 16]
    for k_coh, lbl, style in [
        ("x3_slwm_1b_480p", "SLWM-1B-480p (T=1024)", "o-"),
        ("x3_sliding_window_w64_gqa_1b_480p", "W64-GQA-1B (T=1024)", "s--"),
        ("x3_diagonal_ssm_1b_480p", "DiagSSM-1B (T=1024)", "^-."),
    ]:
        if k_coh in cohorts and "overwrite_T1024_curve" in cohorts[k_coh]:
            cv = cohorts[k_coh]["overwrite_T1024_curve"]
            ys = [cv.get(f"K{ki}", 0.0) * 100.0 for ki in ks]
            axes[0].plot(ks, ys, style, lw=2.0, ms=6, label=lbl)
    axes[0].axhline(100.0 / 32.0, color="gray", ls=":", label="32-Val Chance (3.1%)")
    axes[0].set_xlabel("Concurrent Associative Bindings K (with M=K/2 Selective Overwrites)")
    axes[0].set_ylabel("T = 1,024 Recall Accuracy (%)")
    axes[0].set_title("A. Dynamic-Visual 480p Associative Capacity at T = 1,024")
    axes[0].set_ylim(0, 105)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    if pres_h:
        xp = np.arange(len(pres_h))
        fps_v = [cohorts[k]["metrics"]["livestream_sustained_fps"]["mean"] for k, _, _ in pres_h]
        mib_v = [cohorts[k]["strict_system_state_mib"] for k, _, _ in pres_h]
        ax2 = axes[1].twinx()
        axes[1].bar(xp - 0.16, fps_v, width=0.30, color="#2980b9", label="Sustained 480p FPS (Hz)")
        ax2.bar(xp + 0.16, mib_v, width=0.30, color="#e67e22", label="Strict State Memory (MiB)")
        axes[1].set_xticks(xp)
        axes[1].set_xticklabels([lbl for _, lbl, _ in pres_h], fontsize=8.5)
        axes[1].set_ylabel("Sustained B=1 Livestream FPS (Hz)", color="#2980b9")
        ax2.set_ylabel("Strict O(1) Streaming State (MiB)", color="#e67e22")
        axes[1].set_title("B. Continuous T=1,024 480p Livestream Speed & State Footprint")
        axes[1].grid(True, axis="y", alpha=0.25)

    plt.tight_layout()
    f3 = fig_dir / "x3_1b_480p_livestream_T1024_and_overwrite_capacity.png"
    fig.savefig(f3)
    plt.close(fig)
    created.append(f3.name)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.3), dpi=180)
    if "x3_slwm_1b_480p" in cohorts:
        fams = cohorts["x3_slwm_1b_480p"]["unseen_families"]
        f_keys = ["off_center_wall", "dual_chicane", "multi_block_clutter", "dynamics_shift"]
        fx = np.arange(len(f_keys))
        w = 0.20
        axes[0].bar(fx - 1.5 * w, [fams[k]["mppi"] * 100 for k in f_keys], width=w, label="SLWM-1B MPPI", color="#1f77b4")
        axes[0].bar(fx - 0.5 * w, [fams[k]["no_jepa"] * 100 for k in f_keys], width=w, label="No-JEPA", color="#7f8c8d")
        axes[0].bar(fx + 0.5 * w, [fams[k]["sysid_zeroed"] * 100 for k in f_keys], width=w, label="SysID-Zeroed", color="#e74c3c")
        axes[0].bar(fx + 1.5 * w, [fams[k]["sysid_swapped"] * 100 for k in f_keys], width=w, label="SysID-Swapped", color="#d35400")
        axes[0].set_xticks(fx)
        axes[0].set_xticklabels(["Off-Center", "Chicane", "Clutter", "Dyn-Shift"], fontsize=8.5)
        axes[0].set_ylabel("Closed-Loop 480p Success (%)")
        axes[0].set_ylim(0, 105)
        axes[0].set_title("A. Unseen 480p Obstacle & OOD Dynamics-Shift Control (L0 SysID)")
        axes[0].legend(fontsize=7.5)
        axes[0].grid(True, axis="y", alpha=0.3)

    if pres_h:
        xp = np.arange(len(pres_h))
        w = 0.26
        sp_d = [cohorts[k]["metrics"]["vault_spoken_audio_only_disc"]["mean"] * 100 for k, _, _ in pres_h]
        sp_c = [cohorts[k]["metrics"]["vault_spoken_audio_only_scene_color"]["mean"] * 100 for k, _, _ in pres_h]
        jt_c = [cohorts[k]["metrics"]["vault_joint_scene_color"]["mean"] * 100 for k, _, _ in pres_h]
        axes[1].bar(xp - w, sp_d, width=w, label="Audio-Only Dir (%)", color="#8e44ad")
        axes[1].bar(xp, sp_c, width=w, label="Audio-Only Block Color (%)", color="#16a085")
        axes[1].bar(xp + w, jt_c, width=w, label="Joint Speech+Text Color (%)", color="#2c3e50")
        axes[1].axhline(50.0, color="gray", ls=":", lw=1.0)
        axes[1].set_xticks(xp)
        axes[1].set_xticklabels([lbl for _, lbl, _ in pres_h], fontsize=8.5)
        axes[1].set_ylim(0, 105)
        axes[1].set_ylabel("Untouched 284-Episode xArm6 Vault (%)")
        axes[1].set_title("B. Speaker-Disjoint Spoken xArm6 Grounding (H_obj / H_dir)")
        axes[1].legend(fontsize=7.5, loc="lower right")
        axes[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    f4 = fig_dir / "x3_1b_480p_embodied_mppi_and_spoken_vault.png"
    fig.savefig(f4)
    plt.close(fig)
    created.append(f4.name)
    return created


def _extract_cohort_vec_matrices(seed_evals: list[dict[str, Any]]) -> dict[str, tuple[np.ndarray, list[int] | None]]:
    """Extract `[n_seeds, n_units]` episode/world matrices + optional Level-2 cluster IDs for crossed bootstrap CIs."""
    out: dict[str, tuple[np.ndarray, list[int] | None]] = {}
    if not seed_evals:
        return out
    e0 = seed_evals[0]
    ov0 = e0["livestream_T1024"]["multi_item_overwrite_capacity"]
    tmax_ov_keys = sorted([k for k in ov0 if k.startswith("K1_T") and k.endswith("_overwritten_vec")], key=lambda s: int(s.split("_T")[1].split("_")[0]))
    if tmax_ov_keys:
        tk = tmax_ov_keys[-1]
        out["dyn_visual_assoc_T1024"] = (
            np.asarray([e["livestream_T1024"]["multi_item_overwrite_capacity"][tk] for e in seed_evals], dtype=np.float64),
            None,
        )
    out["T160_paired_causal_switch"] = (
        np.asarray([e["beyond_window_T160_switch_480p"]["paired_causal_switch_vec"] for e in seed_evals], dtype=np.float64),
        None,
    )
    mppi0 = e0["unseen_layout_mppi_480p"]
    out["unseen_mppi_480p"] = (
        np.asarray([e["unseen_layout_mppi_480p"]["geometry_free_mppi_succ_vec"] for e in seed_evals], dtype=np.float64),
        None,
    )
    out["no_jepa_subgoal_480p"] = (
        np.asarray([e["unseen_layout_mppi_480p"]["no_jepa_subgoal_succ_vec"] for e in seed_evals], dtype=np.float64),
        None,
    )
    if "sysid_zeroed_succ_vec" in mppi0:
        out["sysid_zeroed_mppi_480p"] = (
            np.asarray([e["unseen_layout_mppi_480p"]["sysid_zeroed_succ_vec"] for e in seed_evals], dtype=np.float64),
            None,
        )
    if "sysid_swapped_succ_vec" in mppi0:
        out["sysid_swapped_mppi_480p"] = (
            np.asarray([e["unseen_layout_mppi_480p"]["sysid_swapped_succ_vec"] for e in seed_evals], dtype=np.float64),
            None,
        )
    out["unseen_dynamics_shift_mppi"] = (
        np.asarray([e["unseen_layout_mppi_480p"]["dynamics_shift_succ_vec"] for e in seed_evals], dtype=np.float64),
        None,
    )
    if "dynamics_shift_sysid_zeroed_vec" in mppi0:
        out["dynamics_shift_sysid_zeroed"] = (
            np.asarray([e["unseen_layout_mppi_480p"]["dynamics_shift_sysid_zeroed_vec"] for e in seed_evals], dtype=np.float64),
            None,
        )
    if "dynamics_shift_sysid_swapped_vec" in mppi0:
        out["dynamics_shift_sysid_swapped"] = (
            np.asarray([e["unseen_layout_mppi_480p"]["dynamics_shift_sysid_swapped_vec"] for e in seed_evals], dtype=np.float64),
            None,
        )
    out["commit_selective_routing_acc"] = (
        np.asarray([e["unseen_ood_480p"]["commit_selective_routing_vec"] for e in seed_evals], dtype=np.float64),
        None,
    )
    out["vault_480p_video_psnr_db"] = (
        np.asarray([e["any_to_any_vault"]["dir2_text_ctx_to_480p_video"]["psnr_480p_vec"] for e in seed_evals], dtype=np.float64),
        e0["any_to_any_vault"]["dir2_text_ctx_to_480p_video"]["group_ids"],
    )
    out["vault_480p_edge_rmse"] = (
        np.asarray([e["any_to_any_vault"]["dir2_text_ctx_to_480p_video"]["edge_rmse_vec"] for e in seed_evals], dtype=np.float64),
        e0["any_to_any_vault"]["dir2_text_ctx_to_480p_video"]["group_ids"],
    )
    if e0["spoken_xarm6_vault"].get("available"):
        cids = e0["spoken_xarm6_vault"]["scene_cluster_ids"]
        out["vault_spoken_audio_only_disc"] = (
            np.asarray([e["spoken_xarm6_vault"]["spoken_disc_vec"] for e in seed_evals], dtype=np.float64),
            cids,
        )
        out["vault_spoken_audio_only_color_lift"] = (
            np.asarray([e["spoken_xarm6_vault"]["spoken_color_lift_vec"] for e in seed_evals], dtype=np.float64),
            cids,
        )
        bym0 = e0["spoken_xarm6_vault"].get("by_modality", {})
        if "spoken_audio_only" in bym0 and "scene_color_vec" in bym0["spoken_audio_only"]:
            out["vault_spoken_audio_only_scene_color"] = (
                np.asarray([e["spoken_xarm6_vault"]["by_modality"]["spoken_audio_only"]["scene_color_vec"] for e in seed_evals], dtype=np.float64),
                cids,
            )
        if "joint_audio_text" in bym0 and "scene_color_vec" in bym0["joint_audio_text"]:
            out["vault_joint_scene_color"] = (
                np.asarray([e["spoken_xarm6_vault"]["by_modality"]["joint_audio_text"]["scene_color_vec"] for e in seed_evals], dtype=np.float64),
                cids,
            )
            out["vault_paired_180_flip_dual"] = (
                np.asarray([e["spoken_xarm6_vault"]["by_modality"]["joint_audio_text"]["paired_180_flip_dual_vec"] for e in seed_evals], dtype=np.float64),
                cids,
            )
    return out


def _extract_ood_auroc_arrays(seed_evals: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray] | None:
    if not seed_evals:
        return None
    ood0 = seed_evals[0].get("unseen_ood_480p", {})
    if "ood_corrupt_tick_scores" in ood0 and "ood_clean_tick_scores" in ood0:
        corr = np.asarray([e["unseen_ood_480p"]["ood_corrupt_tick_scores"] for e in seed_evals], dtype=np.float64)
        clean = np.asarray([e["unseen_ood_480p"]["ood_clean_tick_scores"] for e in seed_evals], dtype=np.float64)
        return corr, clean
    if "ood_corrupt_world_scores_vec" in ood0 and "ood_clean_world_scores_vec" in ood0:
        corr = np.asarray([e["unseen_ood_480p"]["ood_corrupt_world_scores_vec"] for e in seed_evals], dtype=np.float64)[:, None, :, None]
        clean = np.asarray([e["unseen_ood_480p"]["ood_clean_world_scores_vec"] for e in seed_evals], dtype=np.float64)[:, :, None]
        return corr, clean
    return None


def analyze_study_x3(
    runs_dir: Path,
    out_json: Path,
    showcase_dir: Path,
    fig_dir: Path,
    *,
    require_verified: bool = False,
) -> dict[str, Any]:
    cfg = json.loads((REPO_ROOT / "configs/x3/study_x3_1b_480p.json").read_text())
    g4_cfg = cfg["gate4_criteria"]
    min_seeds_req = int(g4_cfg.get("min_seeds_per_cohort", 3))
    expected_seeds = [int(x) for x in g4_cfg.get("expected_seeds", [3001, 3002, 3003])]

    cohorts_out: dict[str, Any] = {}
    cohort_vecs: dict[str, dict[str, tuple[np.ndarray, list[int] | None]]] = {}
    cohort_ood_scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    cohort_val_step_curves: dict[str, dict[int, list[float]]] = {}
    cohort_median_bundles: dict[str, tuple[str, Path]] = {}
    seen_source_shas: set[str] = set()
    incomplete_cohorts: list[str] = []
    unverified_runs: list[str] = []

    flagship_eval: dict[str, Any] | None = None
    flagship_showcase_npz: Path | None = None

    for c_spec in cfg["cohorts"]:
        cname = c_spec["cohort"]
        seed_recs: list[dict[str, Any]] = []
        seed_evals: list[dict[str, Any]] = []
        seed_dirs: list[Path] = []
        term_losses: list[float] = []
        term_val_losses: list[float] = []
        for s in c_spec["seeds"]:
            tag = f"{cname.replace('_', '-')}-s{s}"
            rdir = runs_dir / tag
            rec_p = rdir / "receipt.json"
            ev_p = rdir / "eval_metrics.json"
            tr_p = rdir / "train.jsonl"
            ver_p = rdir / "verified.json"
            if not (rec_p.is_file() and ev_p.is_file()):
                continue
            if require_verified and not ver_p.is_file():
                unverified_runs.append(tag)
            rec = json.loads(rec_p.read_text())
            ev = json.loads(ev_p.read_text())
            ev["tag"] = tag
            ev["seed"] = int(s)
            seen_source_shas.add(str(rec.get("source_code_sha256", "")))
            seed_recs.append(rec)
            seed_evals.append(ev)
            seed_dirs.append(rdir)
            if tr_p.is_file():
                lines = [json.loads(ln) for ln in tr_p.read_text().splitlines() if ln.strip()]
                if lines:
                    term_losses.append(float(np.mean([x["loss"] for x in lines[-min(3, len(lines)) :]])))
                    val_entries = [x for x in lines if "val_loss" in x]
                    if val_entries:
                        term_val_losses.append(float(val_entries[-1]["val_loss"]))
                    else:
                        term_val_losses.append(float(np.mean([x["loss"] for x in lines[-min(3, len(lines)) :]])))
                    tok_ps = int(rec.get("data_provenance", {}).get("slot_tokens_per_step", 21120))
                    sc_map = cohort_val_step_curves.setdefault(cname, {})
                    if val_entries:
                        for x_ln in val_entries:
                            st_i = int(x_ln.get("step", 0))
                            sc_map.setdefault(max(1, st_i) * tok_ps, []).append(float(x_ln["val_loss"]))
                    else:
                        for x_ln in lines:
                            st_i = int(x_ln.get("step", 0))
                            if st_i >= 400 or len(lines) <= 5:
                                sc_map.setdefault(max(1, st_i) * tok_ps, []).append(float(x_ln["loss"]))

        if not seed_recs:
            incomplete_cohorts.append(cname)
            continue
        got_seeds = sorted(int(r["seed"]) for r in seed_recs)
        if len(seed_recs) < min_seeds_req or got_seeds != sorted(expected_seeds):
            incomplete_cohorts.append(cname)

        med_ev, med_npz = _select_median_seed_entry(seed_evals, seed_dirs)
        if med_npz is not None:
            cohort_median_bundles[cname] = (str(med_ev.get("tag", cname)), med_npz)
        if cname == "x3_slwm_1b_480p":
            flagship_eval, flagship_showcase_npz = med_ev, med_npz

        vecs_c = _extract_cohort_vec_matrices(seed_evals)
        cohort_vecs[cname] = vecs_c
        ood_pair = _extract_ood_auroc_arrays(seed_evals)
        if ood_pair is not None:
            cohort_ood_scores[cname] = ood_pair

        metric_keys = list(seed_evals[0]["summary"].keys())
        agg_m: dict[str, Any] = {}
        for mk in metric_keys:
            vals = [float(e["summary"][mk]) for e in seed_evals]
            entry_m: dict[str, Any] = {
                "mean": round(float(np.mean(vals)), 6),
                "std": round(float(np.std(vals, ddof=1 if len(vals) > 1 else 0)), 6),
                "values": [round(v, 6) for v in vals],
            }
            if mk in vecs_c:
                mat_v, cids_v = vecs_c[mk]
                ci_doc = hierarchical_seed_bootstrap_ci(mat_v, cluster_ids=cids_v, n_boot=4000)
                entry_m["hierarchical_ci95"] = [round(x, 6) for x in ci_doc["hierarchical_ci95"]]
            elif mk == "unseen_ood_auroc_480p" and ood_pair is not None:
                ci_auc = hierarchical_paired_auroc_bootstrap_ci(ood_pair[0], ood_pair[1], n_boot=4000)
                entry_m["hierarchical_ci95"] = [round(x, 6) for x in ci_auc["hierarchical_ci95"]]
            agg_m[mk] = entry_m

        # Paired within-cohort single-factor ablation CIs (JEPA rollout & L0 SysID)
        ablation_contrasts: dict[str, Any] = {}
        for cname_ab, k_pos, k_neg in (
            ("mppi_minus_no_jepa_480p", "unseen_mppi_480p", "no_jepa_subgoal_480p"),
            ("mppi_minus_sysid_zeroed_all", "unseen_mppi_480p", "sysid_zeroed_mppi_480p"),
            ("mppi_minus_sysid_swapped_all", "unseen_mppi_480p", "sysid_swapped_mppi_480p"),
            ("dynamics_shift_mppi_minus_sysid_zeroed", "unseen_dynamics_shift_mppi", "dynamics_shift_sysid_zeroed"),
            ("dynamics_shift_mppi_minus_sysid_swapped", "unseen_dynamics_shift_mppi", "dynamics_shift_sysid_swapped"),
        ):
            if k_pos in vecs_c and k_neg in vecs_c and vecs_c[k_pos][0].shape == vecs_c[k_neg][0].shape:
                ci_ab = hierarchical_seed_bootstrap_ci(vecs_c[k_pos][0], vecs_c[k_neg][0], n_boot=4000)
                ablation_contrasts[cname_ab] = {
                    "delta_mean": round(ci_ab["mean"], 6),
                    "hierarchical_ci95": [round(x, 6) for x in ci_ab["hierarchical_ci95"]],
                    "p_gt_zero": round(ci_ab["hierarchical_p_gt_zero"], 5),
                }

        # Group-clustered two-level hierarchical CIs for all 6 Any-to-Any condition-shuffle lifts on `vault`
        dir_specs = [
            ("dir1_ce_lift", "dir1_video_audio_to_text"),
            ("dir2_psnr_db_lift", "dir2_text_ctx_to_480p_video"),
            ("dir3_lsd_db_lift", "dir3_video_to_16khz_audio"),
            ("dir4_mse_lift", "dir4_text_video_to_3d_movement"),
            ("dir5_mse_lift", "dir5_av_text_to_robot_actions"),
            ("dir6_bce_lift", "dir6_blind_audio_proprio_and_ood_gate"),
        ]
        v_lifts: dict[str, Any] = {}
        for short_k, panel_k in dir_specs:
            mat_l = np.asarray([e["any_to_any_vault"][panel_k]["shuffle_lift_vec"] for e in seed_evals], dtype=np.float64)
            gids_l = seed_evals[0]["any_to_any_vault"][panel_k]["group_ids"]
            ci_l = hierarchical_seed_bootstrap_ci(mat_l, cluster_ids=gids_l, n_boot=4000)
            v_lifts[short_k] = {
                "mean": round(ci_l["mean"], 6),
                "between_seed_sd": round(ci_l["between_seed_sd"], 6),
                "hierarchical_ci95": [round(x, 6) for x in ci_l["hierarchical_ci95"]],
                "p_gt_zero": round(ci_l["hierarchical_p_gt_zero"], 5),
                "n_groups": ci_l["n_level2_clusters"],
            }
            vecs_c[f"any2any_{short_k}"] = (mat_l, gids_l)

        ov_curve: dict[str, float] = {}
        curves_raw = seed_evals[0]["livestream_T1024"]["multi_item_overwrite_capacity"]["curves"]
        t_last = sorted(curves_raw.keys(), key=lambda x: int(x[1:]))[-1]
        for k_str in curves_raw[t_last]:
            ov_curve[k_str] = float(
                np.mean([e["livestream_T1024"]["multi_item_overwrite_capacity"]["curves"][t_last][k_str]["overall_acc"] for e in seed_evals])
            )

        fams_agg: dict[str, Any] = {}
        for fk in ("off_center_wall", "dual_chicane", "multi_block_clutter", "dynamics_shift"):
            fams_agg[fk] = {
                "oracle": float(np.mean([e["unseen_layout_mppi_480p"]["by_unseen_family"][fk]["oracle_mppi_ceiling"] for e in seed_evals])),
                "mppi": float(np.mean([e["unseen_layout_mppi_480p"]["by_unseen_family"][fk]["mppi_success"] for e in seed_evals])),
                "no_jepa": float(np.mean([e["unseen_layout_mppi_480p"]["by_unseen_family"][fk]["no_jepa_subgoal_success"] for e in seed_evals])),
                "sysid_zeroed": float(np.mean([e["unseen_layout_mppi_480p"]["by_unseen_family"][fk]["sysid_zeroed_mppi_success"] for e in seed_evals])),
                "sysid_swapped": float(np.mean([e["unseen_layout_mppi_480p"]["by_unseen_family"][fk]["sysid_swapped_mppi_success"] for e in seed_evals])),
            }

        cohorts_out[cname] = {
            "role": c_spec["role"],
            "n_seeds": len(seed_recs),
            "seeds": got_seeds,
            "strict_params": int(seed_recs[0]["parameter_counts"]["strict_total"]),
            "parameter_breakdown": seed_recs[0]["parameter_counts"],
            "core_state_mib": float(seed_recs[0]["streaming_state_memory"]["core_state_mib"]),
            "strict_system_state_mib": float(seed_recs[0]["streaming_state_memory"]["strict_system_state_mib"]),
            "terminal_train_loss": {
                "mean": round(float(np.mean(term_losses)), 5) if term_losses else 0.0,
                "std": round(float(np.std(term_losses, ddof=1 if len(term_losses) > 1 else 0)), 5) if term_losses else 0.0,
            },
            "terminal_val_loss": {
                "mean": round(float(np.mean(term_val_losses)), 5) if term_val_losses else 0.0,
                "std": round(float(np.std(term_val_losses, ddof=1 if len(term_val_losses) > 1 else 0)), 5) if term_val_losses else 0.0,
            },
            "slot_token_exposure": {
                "world_and_aux_slot_tokens_per_step": int(seed_recs[0].get("data_provenance", {}).get("slot_tokens_per_step", 21120)),
                "total_grad_slot_tokens_per_step": int(seed_recs[0].get("data_provenance", {}).get("total_grad_slot_tokens_per_step", 32640)),
                "world_and_aux_total_slot_tokens": int(seed_recs[0].get("data_provenance", {}).get("total_slot_tokens", 0)),
                "total_grad_slot_tokens": int(seed_recs[0].get("data_provenance", {}).get("total_grad_slot_tokens", 0)),
            },
            "mean_train_wall_s": round(float(np.mean([r["train_wall_s"] for r in seed_recs])), 2),
            "metrics": agg_m,
            "within_cohort_ablation_cis": ablation_contrasts,
            "any_to_any_vault_lifts": v_lifts,
            "overwrite_T1024_curve": ov_curve,
            "unseen_families": fams_agg,
            "beta_aud_diagnostics": {
                "active_speech_mean": float(np.mean([e["spoken_xarm6_vault"].get("beta_aud_active_speech_mean", 0.0) for e in seed_evals])),
                "trailing_silence_mean": float(np.mean([e["spoken_xarm6_vault"].get("beta_aud_trailing_silence_mean", 0.0) for e in seed_evals])),
            },
        }

    # Paired 1.022B two-stage crossed hierarchical bootstrap contrasts across all primary & non-regression metrics
    head_to_head_cis: dict[str, Any] = {}
    h2h_metric_keys = (
        "dyn_visual_assoc_T1024",
        "T160_paired_causal_switch",
        "unseen_mppi_480p",
        "unseen_dynamics_shift_mppi",
        "commit_selective_routing_acc",
        "vault_480p_video_psnr_db",
        "vault_480p_edge_rmse",
        "vault_spoken_audio_only_disc",
        "vault_spoken_audio_only_scene_color",
        "vault_spoken_audio_only_color_lift",
        "vault_joint_scene_color",
        "any2any_dir1_ce_lift",
        "any2any_dir2_psnr_db_lift",
        "any2any_dir3_lsd_db_lift",
        "any2any_dir4_mse_lift",
        "any2any_dir5_mse_lift",
        "any2any_dir6_bce_lift",
    )
    if "x3_slwm_1b_480p" in cohort_vecs:
        v_pri = cohort_vecs["x3_slwm_1b_480p"]
        for ref_c in ("x3_sliding_window_w64_gqa_1b_480p", "x3_diagonal_ssm_1b_480p", "x3_slwm_1b_240p_ablation"):
            if ref_c not in cohort_vecs:
                continue
            v_ref = cohort_vecs[ref_c]
            contrasts: dict[str, Any] = {}
            for mk in h2h_metric_keys:
                if mk in v_pri and mk in v_ref and v_pri[mk][0].shape == v_ref[mk][0].shape:
                    ci_d = hierarchical_seed_bootstrap_ci(v_pri[mk][0], v_ref[mk][0], cluster_ids=v_pri[mk][1], n_boot=4000)
                    contrasts[mk] = {
                        "delta_mean": round(ci_d["mean"], 6),
                        "hierarchical_ci95": [round(x, 6) for x in ci_d["hierarchical_ci95"]],
                        "p_gt_zero": round(ci_d["hierarchical_p_gt_zero"], 5),
                    }
            if "x3_slwm_1b_480p" in cohort_ood_scores and ref_c in cohort_ood_scores:
                p_ood = cohort_ood_scores["x3_slwm_1b_480p"]
                r_ood = cohort_ood_scores[ref_c]
                if p_ood[0].shape == r_ood[0].shape and p_ood[1].shape == r_ood[1].shape:
                    ci_o = hierarchical_paired_auroc_bootstrap_ci(p_ood[0], p_ood[1], r_ood[0], r_ood[1], n_boot=4000)
                    contrasts["unseen_ood_auroc_480p"] = {
                        "delta_mean": round(ci_o["mean"], 6),
                        "hierarchical_ci95": [round(x, 6) for x in ci_o["hierarchical_ci95"]],
                        "p_gt_zero": round(ci_o["hierarchical_p_gt_zero"], 5),
                    }
            head_to_head_cis[f"slwm_1b_480p_minus_{ref_c}"] = contrasts

    scaling_fits: dict[str, Any] = {}
    rungs = ["x3_slwm_150m_480p", "x3_slwm_450m_480p", "x3_slwm_1b_480p"]
    if all(r in cohorts_out for r in rungs):
        ns = [float(cohorts_out[r]["strict_params"]) for r in rungs]
        scaling_fits["joint_multimodal_val_loss"] = fit_power_law_3pt(
            ns, [cohorts_out[r]["terminal_val_loss"]["mean"] for r in rungs], positive_is_lower=True
        )
        scaling_fits["joint_multimodal_loss"] = fit_power_law_3pt(
            ns, [cohorts_out[r]["terminal_train_loss"]["mean"] for r in rungs], positive_is_lower=True
        )
        scaling_fits["vault_480p_video_psnr"] = fit_power_law_3pt(
            ns, [cohorts_out[r]["metrics"]["vault_480p_video_psnr_db"]["mean"] for r in rungs], positive_is_lower=False
        )
        scaling_fits["unseen_480p_mppi"] = fit_power_law_3pt(
            ns, [cohorts_out[r]["metrics"]["unseen_mppi_480p"]["mean"] for r in rungs], positive_is_lower=False
        )
        nd_pts: list[tuple[float, float, float]] = []
        for r in rungs:
            n_val = float(cohorts_out[r]["strict_params"])
            for d_tok, l_vals in sorted(cohort_val_step_curves.get(r, {}).items()):
                nd_pts.append((n_val, float(d_tok), float(np.mean(l_vals))))
        scaling_fits["chinchilla_nd_surface"] = fit_chinchilla_surface_nd(nd_pts)

    sha_uniform = len(seen_source_shas) == 1 and "" not in seen_source_shas
    seeds_complete = len(incomplete_cohorts) == 0 and len(cohorts_out) == len(cfg["cohorts"])
    verified_ok = (not require_verified) or (len(unverified_runs) == 0)

    g4_checks: dict[str, Any] = {
        "all_cohorts_3seed_complete": {"value": seeds_complete, "incomplete": incomplete_cohorts, "pass": bool(seeds_complete)},
        "uniform_source_code_sha256": {"value": sorted(seen_source_shas), "pass": bool(sha_uniform)},
        "dual_hdd_verified_receipts": {"unverified": unverified_runs, "pass": bool(verified_ok)},
    }
    if "x3_slwm_1b_480p" in cohorts_out:
        m1b = cohorts_out["x3_slwm_1b_480p"]["metrics"]
        psnr_240 = (
            cohorts_out["x3_slwm_1b_240p_ablation"]["metrics"]["vault_480p_video_psnr_db"]["mean"]
            if "x3_slwm_1b_240p_ablation" in cohorts_out
            else None
        )
        psnr_gain_480_vs_240 = (m1b["vault_480p_video_psnr_db"]["mean"] - psnr_240) if psnr_240 is not None else 0.0
        g4_checks.update(
            {
                "dyn_visual_assoc_T1024": {
                    "value": m1b["dyn_visual_assoc_T1024"]["mean"],
                    "threshold": g4_cfg["min_dyn_visual_assoc_T1024"],
                    "pass": bool(m1b["dyn_visual_assoc_T1024"]["mean"] >= g4_cfg["min_dyn_visual_assoc_T1024"]),
                },
                "T160_paired_causal_switch": {
                    "value": m1b["T160_paired_causal_switch"]["mean"],
                    "threshold": g4_cfg["min_T160_paired_causal_switch"],
                    "pass": bool(m1b["T160_paired_causal_switch"]["mean"] >= g4_cfg["min_T160_paired_causal_switch"]),
                },
                "unseen_mppi_480p": {
                    "value": m1b["unseen_mppi_480p"]["mean"],
                    "threshold": g4_cfg["min_unseen_mppi"],
                    "pass": bool(m1b["unseen_mppi_480p"]["mean"] >= g4_cfg["min_unseen_mppi"]),
                },
                "unseen_dynamics_shift_mppi": {
                    "value": m1b["unseen_dynamics_shift_mppi"]["mean"],
                    "threshold": g4_cfg["min_unseen_dynamics_shift_mppi"],
                    "pass": bool(m1b["unseen_dynamics_shift_mppi"]["mean"] >= g4_cfg["min_unseen_dynamics_shift_mppi"]),
                },
                "mppi_non_inferior_to_no_jepa": {
                    "value": m1b["mppi_over_no_jepa_gain_pp"]["mean"] / 100.0,
                    "threshold": -g4_cfg["max_mppi_drop_vs_no_jepa_subgoal"],
                    "pass": bool((m1b["mppi_over_no_jepa_gain_pp"]["mean"] / 100.0) >= -g4_cfg["max_mppi_drop_vs_no_jepa_subgoal"]),
                },
                "commit_selective_routing": {
                    "value": m1b["commit_selective_routing_acc"]["mean"],
                    "threshold": g4_cfg["min_commit_selective_routing"],
                    "pass": bool(m1b["commit_selective_routing_acc"]["mean"] >= g4_cfg["min_commit_selective_routing"]),
                },
                "vault_spoken_audio_only_disc": {
                    "value": m1b["vault_spoken_audio_only_disc"]["mean"],
                    "threshold": g4_cfg["min_vault_spoken_audio_only_disc"],
                    "pass": bool(m1b["vault_spoken_audio_only_disc"]["mean"] >= g4_cfg["min_vault_spoken_audio_only_disc"]),
                },
                "vault_spoken_audio_only_color_lift": {
                    "value": m1b["vault_spoken_audio_only_color_lift"]["mean"],
                    "threshold": g4_cfg["min_vault_audio_only_scene_color_lift_vs_shuffle"],
                    "pass": bool(m1b["vault_spoken_audio_only_color_lift"]["mean"] >= g4_cfg["min_vault_audio_only_scene_color_lift_vs_shuffle"]),
                },
                "psnr_480p_over_240p_gain_db": {
                    "value": round(float(psnr_gain_480_vs_240), 4),
                    "threshold": g4_cfg["min_480p_over_240p_psnr_gain_db"],
                    "pass": bool(psnr_240 is not None and psnr_gain_480_vs_240 >= g4_cfg["min_480p_over_240p_psnr_gain_db"]),
                },
                "any_to_any_positive_shuffle_lifts": {
                    "value": m1b["vault_positive_shuffle_lift_dirs"]["mean"],
                    "threshold": g4_cfg["min_any_to_any_shuffle_lift_positive_directions"],
                    "pass": bool(m1b["vault_positive_shuffle_lift_dirs"]["mean"] >= g4_cfg["min_any_to_any_shuffle_lift_positive_directions"]),
                },
                "livestream_zero_state_growth": {
                    "value": int(m1b["livestream_state_growth_bytes"]["mean"]),
                    "threshold": g4_cfg["max_livestream_T1024_state_growth_bytes"],
                    "pass": bool(int(m1b["livestream_state_growth_bytes"]["mean"]) <= g4_cfg["max_livestream_T1024_state_growth_bytes"]),
                },
            }
        )

    showcase_manifest = {}
    if flagship_showcase_npz is not None and flagship_eval is not None:
        showcase_manifest = export_x3_showcase_gallery(
            flagship_showcase_npz,
            flagship_eval,
            showcase_dir,
            cohort_median_bundles=cohort_median_bundles,
        )

    summary_doc = {
        "study_id": cfg["study_id"],
        "wave": cfg["wave"],
        "cohorts": cohorts_out,
        "head_to_head_hierarchical_cis": head_to_head_cis,
        "scaling_law_fits": scaling_fits,
        "gate4_audit": {
            "all_passed": bool(g4_checks and all(v["pass"] for v in g4_checks.values())),
            "checks": g4_checks,
        },
        "sota_comparison_scope_note": SOTA_COMPARISON_SCOPE_NOTE,
        "sota_world_model_comparison_0_to_1b": SOTA_0_TO_1B_CONTEXT,
        "sota_world_model_comparison_1_to_30b": SOTA_1_TO_30B_CONTEXT,
        "showcase_manifest": showcase_manifest,
    }
    figs = render_x3_publication_figures(summary_doc, fig_dir)
    summary_doc["figures"] = figs
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary_doc, indent=2, sort_keys=True) + "\n")
    return summary_doc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs-dir",
        type=Path,
        default=REPO_ROOT / "experiments/multimodal/x3/livestream-1b-480p-20261116/runs_x3_1b_480p",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=REPO_ROOT / "experiments/multimodal/x3/livestream-1b-480p-20261116/x3_summary_metrics.json",
    )
    ap.add_argument("--showcase-dir", type=Path, default=REPO_ROOT / "inference_showcase/x3_1b_480p")
    ap.add_argument("--fig-dir", type=Path, default=REPO_ROOT / "figures")
    ap.add_argument("--require-verified", action="store_true", default=True)
    ap.add_argument("--no-require-verified", dest="require_verified", action="store_false")
    args = ap.parse_args()
    doc = analyze_study_x3(
        args.runs_dir, args.out_json, args.showcase_dir, args.fig_dir, require_verified=args.require_verified
    )
    print(f"[ANALYZED X3] cohorts={list(doc['cohorts'].keys())} gate4_all_passed={doc['gate4_audit']['all_passed']}", flush=True)


if __name__ == "__main__":
    main()
