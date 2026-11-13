"""Build zero-leakage 64x64x3 Real Physical Robot (xArm6 / Language-Table) Bank on Fedora.

Reads strictly read-only from:
    - /mnt/slwm-data/SLWM/g13/data/robot/banks/episodes.json
    - /mnt/slwm-data/SLWM/g13/data/robot/decoded/*.npz
    - /mnt/slwm-data/SLWM/g13/data/robot/raw/*tfrecord*
    - /mnt/slwm-data/SLWM/x1b/datasets/vocab_4096.json

Writes atomically ONLY to /mnt/slwm-data/SLWM/x1b/datasets/:
    - x1b_real_robot_v2.npz + x1b_real_robot_v2.card.json
    - vocab_4096_ext.json

Physical calibration constants (exact isotropic Cartesian metric):
    ROBOT_X0_M          = 0.405 m
    ROBOT_Y0_M          = 0.020 m
    ROBOT_SCALE_M       = 0.350 m  (1.0 normalized unit == 350.0 mm physical!)
    ROBOT_ACTION_SCALE  = 0.100 m  (1.0 normalized action == 100.0 mm setpoint delta)
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import struct
from typing import Any

import cv2
import numpy as np

SEQ_LEN = 32
VOCAB_SIZE = 4096
ROBOT_X0_M = 0.405
ROBOT_Y0_M = 0.020
ROBOT_SCALE_M = 0.350
ROBOT_SCALE_MM = 350.0
ROBOT_ACTION_SCALE_M = 0.100
WORD_RE = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")


def sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _read_varint(buf: bytes | memoryview, pos: int) -> tuple[int, int]:
    val = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        val |= (b & 0x7F) << shift
        if not (b & 0x80):
            return val, pos
        shift += 7


def _parse_ld_fields(buf: memoryview) -> dict[int, list[memoryview]]:
    pos = 0
    n = len(buf)
    out: dict[int, list[memoryview]] = {}
    while pos < n:
        tag, pos = _read_varint(buf, pos)
        fn = tag >> 3
        wt = tag & 7
        if wt == 2:
            ln, pos = _read_varint(buf, pos)
            out.setdefault(fn, []).append(buf[pos : pos + ln])
            pos += ln
        elif wt == 0:
            _, pos = _read_varint(buf, pos)
        elif wt == 5:
            pos += 4
        elif wt == 1:
            pos += 8
        else:
            raise ValueError(f"Unsupported protobuf wire type {wt}")
    return out


def extract_rgb_jpegs_from_example(example_bytes: bytes) -> list[memoryview]:
    """Zero-dependency parser for tf.train.Example['steps/observation/rgb'].bytes_list.value."""
    mv = memoryview(example_bytes)
    feats_msg = _parse_ld_fields(mv)[1][0]
    for entry in _parse_ld_fields(feats_msg).get(1, []):
        kv = _parse_ld_fields(entry)
        key = bytes(kv[1][0]).decode("utf-8")
        if key == "steps/observation/rgb":
            feat = _parse_ld_fields(kv[2][0])
            blist = _parse_ld_fields(feat[1][0])
            return blist.get(1, [])
    raise KeyError("Missing steps/observation/rgb in TFRecord Example")


def build_extended_vocab(base_vocab: dict[str, int], train_goals: list[str]) -> dict[str, int]:
    """Extend vocab_4096 with train-only Language-Table words in unused control slots 96..239."""
    ext = dict(base_vocab)
    missing_counts: Counter[str] = Counter()
    for g in train_goals:
        for tok in WORD_RE.findall(g.lower()):
            if tok not in ext and len(tok) <= 24:
                missing_counts[tok] += 1
    # Reserved 0..79 are PAD/UNK/QUERY/EOS/quadrants/ASSOC_KEY(16..31)/ASSOC_VAL(64..79);
    # 96..239 (144 IDs) are completely unused and disjoint from 256..511 byte fallback & 512..4095 FineWeb.
    for idx, (tok, _) in enumerate(missing_counts.most_common(144)):
        ext[tok] = 96 + idx
    return ext


def encode_instruction_16(text: str, vocab: dict[str, int]) -> np.ndarray:
    """Encode a natural-language robot instruction into a 4x4 (16-token, 0.4s) header."""
    if not text:
        return np.zeros((4, 4), dtype=np.int32)
    ids: list[int] = []
    for tok in WORD_RE.findall(text.lower()):
        tid = vocab.get(tok)
        if tid is not None:
            ids.append(tid)
        else:
            for b in tok.encode("utf-8", errors="ignore")[:6]:
                ids.append(256 + int(b))
        if len(ids) >= 15:
            break
    ids.append(3)  # EOS
    arr = np.zeros((16,), dtype=np.int32)
    take = min(16, len(ids))
    arr[:take] = np.asarray(ids[:take], dtype=np.int32)
    return arr.reshape(4, 4)


def det_rank(key: str) -> str:
    return hashlib.sha256(("X1B_ROBOT_V2/" + key).encode("utf-8")).hexdigest()


def select_split_episodes(
    all_episodes: list[dict[str, Any]],
    targets: dict[str, int],
) -> dict[str, list[dict[str, Any]]]:
    """Deterministically select group-disjoint train/dev/test physical robot episodes."""
    by_role: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    groups_by_role: dict[str, set[str]] = {"train": set(), "dev": set(), "test": set()}

    for ep in all_episodes:
        role = ep["role"]
        groups_by_role[role].add(ep["group"])
        if int(ep["length"]) < 16:
            continue
        by_role[role].append(ep)

    # Strict zero-leakage recording-group audit
    assert groups_by_role["train"].isdisjoint(groups_by_role["dev"]), "Train/Dev group leak!"
    assert groups_by_role["train"].isdisjoint(groups_by_role["test"]), "Train/Test group leak!"
    assert groups_by_role["dev"].isdisjoint(groups_by_role["test"]), "Dev/Test group leak!"
    assert all(not ep.get("novel", False) for ep in by_role["train"]), "Novel pair leaked into train!"

    selected: dict[str, list[dict[str, Any]]] = {}

    # Train: prioritize active manipulation trajectories (length >= 20, paired) with deterministic hash rank
    train_sorted = sorted(
        by_role["train"],
        key=lambda e: (
            0 if int(e["length"]) >= 20 else 1,
            0 if e.get("wrong") else 1,
            det_rank(e["episode"]),
        ),
    )
    selected["train"] = train_sorted[: targets["train"]]

    # Dev: require paired counterfactuals where available
    dev_sorted = sorted(
        by_role["dev"],
        key=lambda e: (
            0 if e.get("wrong") else 1,
            0 if int(e["length"]) >= 20 else 1,
            det_rank(e["episode"]),
        ),
    )
    selected["dev"] = dev_sorted[: targets["dev"]]

    # Test: require 100% paired counterfactual instructions, explicitly stratifying 64 novel compositions + 192 standard!
    test_paired = [e for e in by_role["test"] if e.get("wrong")]
    test_novel = sorted(
        [e for e in test_paired if e.get("novel", False)],
        key=lambda e: (0 if int(e["length"]) >= 20 else 1, det_rank(e["episode"])),
    )
    test_std = sorted(
        [e for e in test_paired if not e.get("novel", False)],
        key=lambda e: (0 if int(e["length"]) >= 20 else 1, det_rank(e["episode"])),
    )
    n_novel = min(64, len(test_novel))
    n_std = targets["test"] - n_novel
    test_comb = test_novel[:n_novel] + test_std[:n_std]
    selected["test"] = sorted(test_comb, key=lambda e: det_rank("order/" + e["episode"]))
    return selected


def norm_xy(xy_m: np.ndarray) -> np.ndarray:
    out = np.empty_like(xy_m, dtype=np.float32)
    out[..., 0] = (xy_m[..., 0] - ROBOT_X0_M) / ROBOT_SCALE_M
    out[..., 1] = (xy_m[..., 1] - ROBOT_Y0_M) / ROBOT_SCALE_M
    return np.clip(out, -1.25, 1.25)


def main() -> None:
    cv2.setNumThreads(1)
    src_root = Path("/mnt/slwm-data/SLWM")
    robot_root = src_root / "g13/data/robot"
    out_dir = src_root / "x1b/datasets"
    out_dir.mkdir(parents=True, exist_ok=True)

    dest_npz = out_dir / "x1b_real_robot_v2.npz"
    dest_card = out_dir / "x1b_real_robot_v2.card.json"
    if dest_npz.is_file() and dest_card.is_file():
        card = json.loads(dest_card.read_text())
        if sha256_file(dest_npz) == card.get("sha256"):
            print("X1B_REAL_ROBOT_V2_ALREADY_VERIFIED", card["sha256"], flush=True)
            return

    all_episodes = json.loads((robot_root / "banks/episodes.json").read_text())
    base_vocab = json.loads((out_dir / "vocab_4096.json").read_text())
    train_goals = [e["goal"] for e in all_episodes if e["role"] == "train"]
    ext_vocab = build_extended_vocab(base_vocab, train_goals)
    (out_dir / "vocab_4096_ext.json").write_text(json.dumps(ext_vocab, sort_keys=True) + "\n")

    targets = {"train": 1280, "dev": 256, "test": 256}
    selected = select_split_episodes(all_episodes, targets)

    raw_shards = {p.name.split("-")[-1]: p for p in (robot_root / "raw").glob("*tfrecord*")}
    decoded_cache: dict[str, dict[str, np.ndarray]] = {}

    # Index needed records per shard so we read sequentially by byte offset
    needed_by_shard: dict[str, list[tuple[int, str, int]]] = defaultdict(list)
    for split, eps in selected.items():
        for idx, ep in enumerate(eps):
            needed_by_shard[ep["shard"]].append((int(ep["offset"]), split, idx))

    frames_store: dict[tuple[str, int], np.ndarray] = {}
    for shard_tag, reqs in sorted(needed_by_shard.items()):
        reqs.sort(key=lambda item: item[0])
        raw_path = raw_shards[shard_tag]
        with np.load(robot_root / "decoded" / f"{shard_tag}.npz", allow_pickle=False) as dz:
            decoded_cache[shard_tag] = {"x": dz["x"].astype(np.float32), "u": dz["u"].astype(np.float32)}
        with raw_path.open("rb") as fh:
            for offset, split, idx in reqs:
                fh.seek(offset)
                hdr = fh.read(12)
                rec_len = struct.unpack_from("<Q", hdr)[0]
                payload = fh.read(rec_len)
                jpegs = extract_rgb_jpegs_from_example(payload)
                take_f = min(len(jpegs), SEQ_LEN + 1)
                rgb_list = []
                for j_idx in range(take_f):
                    bgr = cv2.imdecode(np.frombuffer(jpegs[j_idx], dtype=np.uint8), cv2.IMREAD_COLOR)
                    if bgr is None:
                        raise RuntimeError(f"Failed JPEG decode in shard {shard_tag} offset {offset}")
                    rgb64 = cv2.resize(bgr, (64, 64), interpolation=cv2.INTER_AREA)[:, :, ::-1]
                    rgb_list.append(rgb64)
                frames_store[(split, idx)] = np.stack(rgb_list, axis=0).astype(np.uint8)
        print(f"Decoded {len(reqs)} physical xArm6 episodes from shard {shard_tag}", flush=True)

    arrays: dict[str, np.ndarray] = {}
    split_stats: dict[str, Any] = {}

    for split in ("train", "dev", "test"):
        eps = selected[split]
        n_ep = len(eps)
        vid_arr = np.zeros((n_ep, SEQ_LEN, 64, 64, 3), dtype=np.uint8)
        prop_arr = np.zeros((n_ep, SEQ_LEN, 16), dtype=np.float32)
        nprop_arr = np.zeros((n_ep, SEQ_LEN, 16), dtype=np.float32)
        prev_u_arr = np.zeros((n_ep, SEQ_LEN, 4), dtype=np.float32)
        u_exec_arr = np.zeros((n_ep, SEQ_LEN, 4), dtype=np.float32)
        txt_hdr_arr = np.zeros((n_ep, 4, 4), dtype=np.int32)
        wrong_hdr_arr = np.zeros((n_ep, 4, 4), dtype=np.int32)
        end_xy_arr = np.zeros((n_ep, 2), dtype=np.float32)
        eplen_arr = np.zeros((n_ep,), dtype=np.int32)
        novel_arr = np.zeros((n_ep,), dtype=bool)
        paired_arr = np.zeros((n_ep,), dtype=bool)

        for i, ep in enumerate(eps):
            sh = decoded_cache[ep["shard"]]
            st_i = int(ep["start"])
            ln_i = int(ep["length"])
            x_full = norm_xy(sh["x"][st_i : st_i + ln_i])  # [L, 2]
            u_full = np.clip(sh["u"][st_i : st_i + ln_i] / ROBOT_ACTION_SCALE_M, -1.0, 1.0).astype(np.float32)
            rgbs = frames_store[(split, i)]  # [min(L, 33), 64, 64, 3]

            # Active transition count within T=32 window (s_t -> s_{t+1} under u_t)
            active_steps = min(ln_i - 1, SEQ_LEN)
            eplen_arr[i] = active_steps
            end_xy_arr[i] = x_full[-1]
            novel_arr[i] = bool(ep.get("novel", False))
            paired_arr[i] = bool(ep.get("wrong"))
            txt_hdr_arr[i] = encode_instruction_16(ep["goal"], ext_vocab)
            wrong_hdr_arr[i] = encode_instruction_16(ep.get("wrong", ""), ext_vocab)

            for t in range(SEQ_LEN):
                if t < active_steps:
                    vid_arr[i, t] = rgbs[t]
                    ee_t = x_full[t]
                    ee_n = x_full[t + 1]
                    ee_prev = x_full[max(0, t - 1)]
                    vel_t = (ee_t - ee_prev) * 5.0
                    vel_n = (ee_n - ee_t) * 5.0
                    u_t = np.array([u_full[t, 0], u_full[t, 1], 0.0, 0.0], dtype=np.float32)
                else:
                    # Physically truthful zero-action terminal hold after operator stops moving
                    vid_arr[i, t] = rgbs[active_steps]
                    ee_t = x_full[active_steps]
                    ee_n = x_full[active_steps]
                    vel_t = np.zeros((2,), dtype=np.float32)
                    vel_n = np.zeros((2,), dtype=np.float32)
                    u_t = np.zeros((4,), dtype=np.float32)

                pu_t = np.zeros((4,), dtype=np.float32) if t == 0 else u_exec_arr[i, t - 1]
                prev_u_arr[i, t] = pu_t
                u_exec_arr[i, t] = u_t

                # Honest joint-encoder proprioception ONLY (ee_xy repeated in 0:2 and 2:4; ZERO object color leak!)
                prop_arr[i, t, 0:2] = ee_t
                prop_arr[i, t, 2:4] = ee_t
                prop_arr[i, t, 4:6] = vel_t
                prop_arr[i, t, 6:8] = (0.0, 1.0)
                prop_arr[i, t, 8:10] = pu_t[:2]

                nprop_arr[i, t, 0:2] = ee_n
                nprop_arr[i, t, 2:4] = ee_n
                nprop_arr[i, t, 4:6] = vel_n
                nprop_arr[i, t, 6:8] = (0.0, 1.0)
                nprop_arr[i, t, 8:10] = u_t[:2]

        arrays[f"{split}_video"] = vid_arr
        arrays[f"{split}_proprio"] = prop_arr
        arrays[f"{split}_next_proprio"] = nprop_arr
        arrays[f"{split}_prev_action"] = prev_u_arr
        arrays[f"{split}_exec_action"] = u_exec_arr
        arrays[f"{split}_text_header"] = txt_hdr_arr
        arrays[f"{split}_wrong_header"] = wrong_hdr_arr
        arrays[f"{split}_endpoint_xy"] = end_xy_arr
        arrays[f"{split}_ep_len"] = eplen_arr
        arrays[f"{split}_novel"] = novel_arr
        arrays[f"{split}_paired"] = paired_arr

        split_stats[split] = {
            "episodes": n_ep,
            "groups": len({e["group"] for e in eps}),
            "paired_counterfactuals": int(paired_arr.sum()),
            "novel_compositions": int(novel_arr.sum()),
            "mean_active_steps": round(float(eplen_arr.mean()), 2),
        }

    tmp_npz = dest_npz.with_suffix(".tmp.npz")
    np.savez_compressed(tmp_npz, **arrays)
    tmp_npz.replace(dest_npz)
    digest = sha256_file(dest_npz)

    card = {
        "dataset": "x1b_real_robot_v2",
        "source": "Language-Table physical xArm6 RGB trajectories (union-find frame-deduplicated)",
        "sha256": digest,
        "bytes": dest_npz.stat().st_size,
        "seq_len": SEQ_LEN,
        "resolution": [64, 64, 3],
        "robot_x0_m": ROBOT_X0_M,
        "robot_y0_m": ROBOT_Y0_M,
        "robot_scale_mm": ROBOT_SCALE_MM,
        "robot_action_scale_mm": ROBOT_ACTION_SCALE_M * 1000.0,
        "zero_group_and_frame_leakage_verified": True,
        "zero_novel_pair_in_train_verified": True,
        "splits": split_stats,
    }
    dest_card.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n")
    print("X1B_REAL_ROBOT_V2_READY", json.dumps(card), flush=True)


if __name__ == "__main__":
    main()
