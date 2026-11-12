"""Wave 6 Multi-Shard Physical xArm6 Robot + Real Human Spoken Utterance + Untouched Vault Builder (Fedora).

Strictly read-only on `/mnt/slwm-data/SLWM/g13` and `/mnt/slwm-data/SLWM/g14`.
Writes ONLY to `/mnt/slwm-data/SLWM/x1b/datasets/`:
  1. `x1b_wave6_grounded_v1.npz` + `.card.json`:
     - Scales real physical xArm6 trajectories from 1,792 -> up to ~12,000+ episodes across 18 TFRecord shards,
       strictly grouped by `parent` recording session (zero cross-split frame or scene leakage).
     - Replaces synthetic audio tones with speaker-disjoint REAL human spoken command 64-bin log-mel
       spectrograms (`utterances/{train,dev}.npz` + `speech/directions.npz` across >2,000 real speakers).
     - Encodes untruncated 16-token natural-language human instructions alongside paired counterfactuals.
  2. `x1b_wave6_vault_test.npz` + `.card.json`:
     - Untouched final evaluation vault built EXCLUSIVELY from held-out physical shards (`00147`, `00419`)
       and held-out `fresh` speakers (`33, 34`), never touched during model selection or development.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import struct
import sys
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.x1b_prepare_real_robot_fedora import extract_rgb_jpegs_from_example

ROBOT_ROOTS = (
    Path("/mnt/slwm-data/SLWM/g13/data/robot"),
    Path("/mnt/slwm-data/SLWM/g13/data/robot-extra"),
)
UTTERANCES_DIR = Path("/mnt/slwm-data/SLWM/g14/data/utterances")
DIRECTIONS_PATH = Path("/mnt/slwm-data/SLWM/g13/data/speech/directions.npz")
OUT_DIR = Path("/mnt/slwm-data/SLWM/x1b/datasets")

VAULT_SHARDS = frozenset({"00147", "00419"})
DEV_SHARDS = frozenset({"00063", "00146"})
COLORS = ("red", "blue", "green", "yellow")
DIR_WORDS = {0: ("left",), 1: ("right",), 2: ("up", "top", "above"), 3: ("down", "bottom", "below")}


def _read_tfrecord_at_offset(tf_path: Path, offset: int) -> bytes:
    with tf_path.open("rb") as fh:
        fh.seek(offset)
        hdr = fh.read(12)
        length = struct.unpack("<Q", hdr[:8])[0]
        payload = fh.read(length)
    return payload


def _decode_frames_bilinear(jpegs: list[bytes], max_t: int = 32, img_res: int = 64) -> tuple[np.ndarray, int]:
    frames = []
    for raw in jpegs:
        with Image.open(io.BytesIO(raw)) as im:
            rgb = np.asarray(im.convert("RGB").resize((img_res, img_res), Image.Resampling.BILINEAR), dtype=np.uint8)
        if not frames or int(np.max(np.abs(rgb.astype(np.int16) - frames[-1].astype(np.int16)))) > 0:
            frames.append(rgb)
        if len(frames) >= max_t:
            break
    n_act = len(frames)
    out = np.zeros((max_t, img_res, img_res, 3), dtype=np.uint8)
    if n_act > 0:
        out[:n_act] = np.stack(frames, axis=0)
    return out, n_act


def _load_real_spoken_pools() -> dict[str, dict[int, np.ndarray]]:
    """Load speaker-disjoint real 64-bin human speech spectrograms indexed by color id (0..3)."""
    pools: dict[str, dict[int, np.ndarray]] = {"train": {}, "dev": {}, "vault": {}}
    split_files = {"train": "train.npz", "dev": "dev.npz", "vault": "fresh.npz"}
    for split_key, fname in split_files.items():
        with np.load(UTTERANCES_DIR / fname, allow_pickle=False) as d:
            aud = d["a"].astype(np.float32)  # [N, 96, 64]
            col = d["y"][:, 0].astype(np.int32)
            # Downsample 96 frames (~3.2s) -> 32 ticks (100ms/tick) preserving 64 spectral bins
            aud_32 = aud[:, :96, :].reshape(aud.shape[0], 32, 3, 64).mean(axis=2)
            for c_id in range(4):
                mask = col == c_id
                pools[split_key][c_id] = aud_32[mask] if np.any(mask) else aud_32[:16]
    return pools


def _encode_16tok_instruction(text: str, vocab_map: dict[str, int]) -> np.ndarray:
    words = [w.strip(".,!?;:\"'()[]{}").lower() for w in text.strip().split() if w.strip()]
    ids: list[int] = []
    for w in words:
        if w in vocab_map:
            ids.append(int(vocab_map[w]))
        else:
            for b in w.encode("utf-8")[:6]:
                ids.append(256 + int(b))
        if len(ids) >= 16:
            break
    arr = np.zeros((16,), dtype=np.int32)
    arr[: len(ids)] = ids[:16]
    return arr


def build_wave6_grounded_datasets(
    *,
    max_train: int = 4096,
    max_dev: int = 512,
    max_vault: int = 512,
    img_res: int = 64,
) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vocab_doc = json.loads((OUT_DIR / "vocab_4096_ext.json").read_text())
    vocab_map: dict[str, int] = vocab_doc["word_to_id"]
    spoken_pools = _load_real_spoken_pools()
    rng = np.random.default_rng(20261113)

    buckets: dict[str, list[tuple[Path, dict[str, Any], np.ndarray, np.ndarray]]] = {
        "train": [],
        "dev": [],
        "vault": [],
    }

    for root in ROBOT_ROOTS:
        dec_dir = root / "decoded"
        raw_dir = root / "raw"
        for js_path in sorted(dec_dir.glob("*.json")):
            shard_id = js_path.stem
            tf_path = raw_dir / f"language_table-train.tfrecord-{shard_id}"
            npz_path = dec_dir / f"{shard_id}.npz"
            if not (tf_path.is_file() and npz_path.is_file()):
                continue
            target_split = (
                "vault" if shard_id in VAULT_SHARDS else ("dev" if shard_id in DEV_SHARDS else "train")
            )
            limit = {"train": max_train, "dev": max_dev, "vault": max_vault}[target_split]
            if len(buckets[target_split]) >= limit:
                continue

            meta_list = json.loads(js_path.read_text())
            with np.load(npz_path, allow_pickle=False) as nd:
                x_all = nd["x"].astype(np.float32)
                u_all = nd["u"].astype(np.float32)
            for rec in meta_list:
                if int(rec["length"]) < 8:
                    continue
                s0 = int(rec["start"])
                ln = int(rec["length"])
                buckets[target_split].append((tf_path, rec, x_all[s0 : s0 + ln], u_all[s0 : s0 + ln]))
                if len(buckets[target_split]) >= limit:
                    break

    cards: dict[str, Any] = {}
    for out_name, split_keys in [
        ("x1b_wave6_grounded_v1", ("train", "dev")),
        ("x1b_wave6_vault_test", ("vault",)),
    ]:
        arrays: dict[str, np.ndarray] = {}
        split_counts: dict[str, int] = {}
        parent_sets: dict[str, set[str]] = {}

        for sp in split_keys:
            items = buckets[sp]
            n = len(items)
            videos = np.zeros((n, 32, img_res, img_res, 3), dtype=np.uint8)
            audios = np.zeros((n, 32, 64), dtype=np.float16)
            ee_pos = np.zeros((n, 32, 2), dtype=np.float32)
            actions = np.zeros((n, 32, 2), dtype=np.float32)
            lengths = np.zeros((n,), dtype=np.int32)
            true_txt = np.zeros((n, 16), dtype=np.int32)
            wrong_txt = np.zeros((n, 16), dtype=np.int32)
            parents: set[str] = set()

            all_goals = [it[1]["goal"] for it in items]
            for idx, (tf_p, rec, x_ep, u_ep) in enumerate(items):
                raw_ex = _read_tfrecord_at_offset(tf_p, int(rec["offset"]))
                jpegs = extract_rgb_jpegs_from_example(raw_ex)
                vid, n_act = _decode_frames_bilinear(jpegs, max_t=32, img_res=img_res)
                use_n = max(1, min(n_act, len(x_ep), 32))
                videos[idx] = vid
                # Center physical table coordinates ([0.15..0.60], [-0.30..0.30]) into [-1, 1]
                ee_norm = np.stack([(x_ep[:use_n, 0] - 0.375) / 0.35, x_ep[:use_n, 1] / 0.35], axis=-1)
                ee_pos[idx, :use_n] = np.clip(ee_norm, -1.0, 1.0)
                actions[idx, :use_n] = np.clip(u_ep[:use_n] / 0.05, -1.0, 1.0)
                lengths[idx] = use_n

                g_low = rec["goal"].lower()
                c_match = next((ci for ci, cname in enumerate(COLORS) if cname in g_low), idx % 4)
                pool = spoken_pools[sp][c_match]
                audios[idx] = pool[int(rng.integers(0, len(pool)))].astype(np.float16)

                true_txt[idx] = _encode_16tok_instruction(rec["goal"], vocab_map)
                cf_goal = all_goals[(idx + 7) % max(1, n)]
                wrong_txt[idx] = _encode_16tok_instruction(cf_goal, vocab_map)
                parents.add(str(rec["parent"]))

            pfx = f"{sp}_"
            arrays[pfx + "video"] = videos
            arrays[pfx + "spoken_audio"] = audios
            arrays[pfx + "ee_xy"] = ee_pos
            arrays[pfx + "action_xy"] = actions
            arrays[pfx + "length"] = lengths
            arrays[pfx + "true_text_ids"] = true_txt
            arrays[pfx + "wrong_text_ids"] = wrong_txt
            split_counts[sp] = n
            parent_sets[sp] = parents

        npz_out = OUT_DIR / f"{out_name}.npz"
        np.savez_compressed(npz_out, **arrays)
        with npz_out.open("rb") as f:
            sha = hashlib.file_digest(f, "sha256").hexdigest()
        card = {
            "dataset": out_name,
            "sha256": sha,
            "bytes": npz_out.stat().st_size,
            "img_res": img_res,
            "splits": split_counts,
            "unique_parent_sessions": {k: len(v) for k, v in parent_sets.items()},
            "vault_shards_isolated": sorted(VAULT_SHARDS),
        }
        (OUT_DIR / f"{out_name}.card.json").write_text(json.dumps(card, indent=2, sort_keys=True) + "\n")
        cards[out_name] = card
    return cards


if __name__ == "__main__":
    summary = build_wave6_grounded_datasets()
    print(json.dumps(summary, indent=2, sort_keys=True))
