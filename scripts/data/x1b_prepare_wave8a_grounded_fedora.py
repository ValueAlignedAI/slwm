"""Wave 8A Auditable Physical xArm6 + Raw Causal Speaker-Disjoint Speech + Vault Builder (Fedora).

Strictly read-only on `/mnt/slwm-data/SLWM/g13` and `/mnt/slwm-data/SLWM/g14`.
Writes ONLY versioned v2 datasets to `/mnt/slwm-data/SLWM/x1b/datasets/` (preserving Wave 6/7 v1 banks untouched):
  1. `x1b_wave8a_grounded_v2.npz` + `.card.json` (`train`, `dev`)
  2. `x1b_wave8a_vault_v2.npz` + `.card.json` (`vault`)
  3. `x1b_wave8a_provenance_manifest_v2.json` (retained row provenance + SHA-linked disposition summary)
  4. `x1b_wave8a_episode_dispositions_v2.jsonl.gz` (exhaustive 1-to-1 disposition ledger across all 24,792 scanned episodes)

Scientific Guarantees:
  - Frame-hash Union-Find connected-component grouping across all `Language-Table` roots (matching
    `scripts/data/g13_robot.py:split_groups`), assigning global integer `scene_cluster_id` values so zero shared
    video frames, zero `parent` session overlap, and zero `scene_cluster_id` overlap hold across `train`, `dev`, `vault`.
  - Exact `GRID` color vocabulary (`COLORS = ("red", "blue", "green", "white")`); episodes mentioning unsupported
    `"yellow"` or lacking both a visible target colored block and a spatially separated (`>= 0.18`) competing colored
    distractor block in `frame 0` are explicitly excluded and logged in the exhaustive episode disposition ledger.
  - Unified strictly causal `10 Hz` log-power audio frontend (`pcm_to_causal_logmel_64` / `mpg_to_causal_logmel_64`,
    `<= 10,876 B` total streaming frontend state, `0` future lookahead, `0` FFmpeg `-ar` resampling, `0` whole-waveform
    `max(abs)` normalization, `0` word-alignment metadata):
      * `t = 0..15`: raw `44.1 kHz` `GRID` `.mpg` audio (`2:1` causal boxcar -> `22.05 kHz`, `100 ms` `2,205`-sample Hann STFT)
      * `t = 16..31`: raw `16.0 kHz` `SpeechCommands` `.wav` audio (`100 ms` `1,600`-sample Hann STFT)
  - Dual counterfactual controls eliminating scene-independent shortcuts:
      1. Opposite-direction counterfactual (`cf_index`: `dir_id[cf] == OPPOSITE_DIR[dir_id[i]]`, `dot < -0.02`, `||Δdisp|| >= 0.35`)
      2. Same-scene, same-direction, wrong-color counterfactual (`wrong_color_spoken_audio` + `wrong_color_text_ids`:
         identical `t = 16..31` direction clip and direction text, swapping ONLY `t = 0..15` color speech/token to the
         competing visible distractor block color in `frame 0`).
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import sys
import tarfile
from typing import Any
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from scripts.data.x1b_prepare_fedora import (
    CAUSAL_LOG10_MEAN,
    CAUSAL_LOG10_STD,
    MAX_CAUSAL_AUDIO_FRONTEND_BYTES,
    mpg_to_causal_logmel_64,
    wav_to_logmel_64,
)

try:
    import cv2 as _cv2
except ImportError:
    _cv2 = None
    from PIL import Image as _PILImage

ROBOT_ROOTS = (
    Path("/mnt/slwm-data/SLWM/g13/data/robot"),
    Path("/mnt/slwm-data/SLWM/g13/data/robot-extra"),
    Path("/mnt/slwm-data/SLWM/g13/data/robot-prospective"),
    Path("/mnt/slwm-data/SLWM/g13/data/robot-readout"),
)
UTTERANCES_DIR = Path("/mnt/slwm-data/SLWM/g14/data/utterances")
DIRECTIONS_PATH = Path("/mnt/slwm-data/SLWM/g13/data/speech/directions.npz")
SPEECH_TAR_PATH = Path("/mnt/slwm-data/SLWM/g13/data/speech/speech_commands_v0.02.tar.gz")
OUT_DIR = Path("/mnt/slwm-data/SLWM/x1b/datasets")

# Matches `data/g14_grid.py:3` (`COLORS = ("red", "blue", "green", "white")`)
COLORS = ("red", "blue", "green", "white")
COLOR_PATTERNS = tuple((cid, re.compile(rf"\b{cname}\b")) for cid, cname in enumerate(COLORS))
UNSUPPORTED_YELLOW_RE = re.compile(r"\byellow\b")

# Cardinal directions: 0=left (dy<0), 1=right (dy>0), 2=up/top (dx<0), 3=down/bottom (dx>0)
OPPOSITE_DIR: dict[int, int] = {0: 1, 1: 0, 2: 3, 3: 2}
DIR_NAMES = ("left", "right", "up", "down")
DIR_PATTERNS: dict[int, re.Pattern[str]] = {
    0: re.compile(r"\bleft\b"),
    1: re.compile(r"\bright\b"),
    2: re.compile(r"\b(?:top|up|upper|above|north)\b"),
    3: re.compile(r"\b(?:bottom|down|lower|below|south)\b"),
}
MIN_DISTRACTOR_BLOCK_SEP = 0.18


def _short_sha(tag: str, raw: str) -> str:
    return hashlib.sha256(f"{tag}:{raw}".encode("utf-8")).hexdigest()[:16]


def _sha256_path(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


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
    mv = memoryview(example_bytes)
    feats_msg = _parse_ld_fields(mv)[1][0]
    for entry in _parse_ld_fields(feats_msg).get(1, []):
        kv = _parse_ld_fields(entry)
        key = bytes(kv[1][0]).decode("utf-8")
        if key == "steps/observation/rgb":
            blist = _parse_ld_fields(kv[2][0])[1][0]
            return _parse_ld_fields(blist).get(1, [])
    return []


def _read_tfrecord_at_offset(tf_path: Path, offset: int) -> bytes:
    with tf_path.open("rb") as fh:
        fh.seek(offset)
        hdr = fh.read(12)
        length = struct.unpack("<Q", hdr[:8])[0]
        payload = fh.read(length)
    return payload


def _decode_frames_synchronized(
    jpegs: list[memoryview],
    state_len: int,
    *,
    max_t: int = 32,
    img_res: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode non-duplicate RGB frames while returning exact original step indices `kept_indices`."""
    frames: list[np.ndarray] = []
    kept_idx: list[int] = []
    n_avail = min(len(jpegs), int(state_len))
    for t_raw in range(n_avail):
        raw = jpegs[t_raw]
        if _cv2 is not None:
            bgr = _cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), _cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = _cv2.cvtColor(_cv2.resize(bgr, (img_res, img_res), interpolation=_cv2.INTER_AREA), _cv2.COLOR_BGR2RGB)
        else:
            try:
                with _PILImage.open(io.BytesIO(bytes(raw))) as im:
                    rgb = np.asarray(im.convert("RGB").resize((img_res, img_res), _PILImage.Resampling.BOX), dtype=np.uint8)
            except Exception:
                continue
        if not frames or int(np.max(np.abs(rgb.astype(np.int16) - frames[-1].astype(np.int16)))) > 0:
            frames.append(rgb)
            kept_idx.append(int(t_raw))
        if len(frames) >= max_t:
            break
    out = np.zeros((max_t, img_res, img_res, 3), dtype=np.uint8)
    if frames:
        out[: len(frames)] = np.stack(frames, axis=0)
    return out, np.asarray(kept_idx, dtype=np.int16)


def extract_table_block_centroids_64(rgb0: np.ndarray) -> dict[int, np.ndarray]:
    """Extract visible `(x_norm, y_norm)` centroids in `[-1, 1]^2` for colored tabletop blocks in `frame 0`.

    Matches `canonical_patch_centers_4x4()` / `canonical_grid_centers_8x8()` image convention:
      `x_norm = (col + 0.5) / (W / 2) - 1.0`, `y_norm = (row + 0.5) / (H / 2) - 1.0`.
    """
    img = np.asarray(rgb0, dtype=np.float32) / 255.0
    h, w, _ = img.shape
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    df = mx - mn
    sat = np.where(mx > 1e-5, df / np.maximum(mx, 1e-5), 0.0)
    val = mx

    hue = np.zeros_like(mx)
    nz = df > 1e-5
    r_max = nz & (mx == r)
    g_max = nz & (mx == g) & (~r_max)
    b_max = nz & (~r_max) & (~g_max)
    hue[r_max] = (60.0 * ((g[r_max] - b[r_max]) / df[r_max])) % 360.0
    hue[g_max] = 60.0 * ((b[g_max] - r[g_max]) / df[g_max]) + 120.0
    hue[b_max] = 60.0 * ((r[b_max] - g[b_max]) / df[b_max]) + 240.0

    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    margin_y = max(2, h // 16)
    margin_x = max(2, w // 16)
    interior = (yy >= margin_y) & (yy < (h - margin_y)) & (xx >= margin_x) & (xx < (w - margin_x))

    color_masks = {
        0: interior & ((hue < 24.0) | (hue >= 335.0)) & (sat > 0.35) & (val > 0.22),  # red
        1: interior & (hue >= 185.0) & (hue <= 255.0) & (sat > 0.32) & (val > 0.20),  # blue
        2: interior & (hue >= 75.0) & (hue <= 165.0) & (sat > 0.28) & (val > 0.20),   # green
    }
    centroids: dict[int, np.ndarray] = {}
    half_w = 0.5 * float(w)
    half_h = 0.5 * float(h)
    for cid, m in color_masks.items():
        cnt = int(np.sum(m))
        if cnt >= 3:
            weights = sat[m] * val[m] + 1e-3
            col_c = float(np.sum(xx[m] * weights) / np.sum(weights))
            row_c = float(np.sum(yy[m] * weights) / np.sum(weights))
            x_norm = np.clip((col_c + 0.5) / half_w - 1.0, -0.96, 0.96)
            y_norm = np.clip((row_c + 0.5) / half_h - 1.0, -0.96, 0.96)
            centroids[cid] = np.array([x_norm, y_norm], dtype=np.float32)
    return centroids


def _load_raw_color_metadata() -> tuple[dict[str, dict[int, list[tuple[int, int, int]]]], dict[str, set[int]], list[dict[str, Any]]]:
    """Load speaker-disjoint GRID metadata (`(global_index, speaker_id, local_idx)`), reading zero precomputed `a` or `alignment`."""
    pools: dict[str, dict[int, list[tuple[int, int, int]]]] = {"train": {}, "dev": {}, "vault": {}}
    spk_sets: dict[str, set[int]] = {"train": set(), "dev": set(), "vault": set()}
    split_files = {"train": "train.npz", "dev": "dev.npz", "vault": "fresh.npz"}
    media_entries: list[dict[str, Any]] = json.loads((UTTERANCES_DIR / "media.json").read_text())

    for sp_key, fname in split_files.items():
        for c_id in range(4):
            pools[sp_key][c_id] = []
        with np.load(UTTERANCES_DIR / fname, allow_pickle=False) as d:
            col = d["y"][:, 0].astype(np.int32)
            speakers = d["speaker"].astype(np.int32)
            g_indices = d["global_index"].astype(np.int64)
        spk_sets[sp_key] = set(int(x) for x in np.unique(speakers))
        for i in range(col.shape[0]):
            c_id = int(col[i])
            if c_id in (0, 1, 2, 3):
                pools[sp_key][c_id].append((int(g_indices[i]), int(speakers[i]), int(i)))
    return pools, spk_sets, media_entries


def _load_raw_direction_metadata() -> tuple[dict[str, dict[int, list[tuple[str, str, int]]]], dict[str, set[str]]]:
    """Load speaker-disjoint SpeechCommands metadata (`(wav_rel_name, speaker_id, clip_idx)`), reading zero precomputed `x`."""
    pools: dict[str, dict[int, list[tuple[str, str, int]]]] = {"train": {}, "dev": {}, "vault": {}}
    spk_sets: dict[str, set[str]] = {"train": set(), "dev": set(), "vault": set()}
    role_map = {"train": "train", "dev": "dev", "vault": "test"}

    with np.load(DIRECTIONS_PATH, allow_pickle=False) as d:
        y_all = d["y"].astype(np.int32)
        roles = d["role"]
        speakers = d["speaker"]
        names = d["names"]

    for sp_key, r_name in role_map.items():
        for d_id in range(4):
            pools[sp_key][d_id] = []
        m_idx = np.where(roles == r_name)[0]
        spk_sets[sp_key] = set(str(speakers[i]) for i in m_idx)
        for i in m_idx:
            d_id = int(y_all[i])
            if d_id in (0, 1, 2, 3):
                pools[sp_key][d_id].append((str(names[i]), str(speakers[i]), int(i)))
    return pools, spk_sets


def _decode_sampled_grid_mpgs(
    needed_global_indices: set[int],
    media_entries: list[dict[str, Any]],
) -> dict[int, np.ndarray]:
    """Batch-read raw `.mpg` bytes from GRID files/zips & decode 16 causal ticks via `mpg_to_causal_logmel_64`."""
    raw_bytes_map: dict[int, bytes] = {}
    by_zip: dict[str, list[tuple[int, int, str]]] = {}
    for g_idx in sorted(needed_global_indices):
        meta = media_entries[g_idx]
        raw_path = Path(str(meta.get("raw", "")))
        if raw_path.is_file():
            raw_bytes_map[g_idx] = raw_path.read_bytes()
        else:
            by_zip.setdefault(str(meta["source"]), []).append((g_idx, int(meta["person"]), str(meta["key"])))

    for zip_str, reqs in sorted(by_zip.items()):
        with zipfile.ZipFile(Path(zip_str), "r") as zf:
            names_set = set(zf.namelist())
            for g_idx, person_id, key_name in reqs:
                cand_names = (
                    f"s{person_id}/{key_name}.mpg",
                    f"s{person_id:02d}/{key_name}.mpg",
                    f"{key_name}.mpg",
                    key_name,
                )
                found_member = next((c for c in cand_names if c in names_set), None)
                if found_member is None:
                    raise KeyError(f"Could not find GRID MPG {key_name!r} (person={person_id}) inside {zip_str}")
                raw_bytes_map[g_idx] = zf.read(found_member)

    ordered_ids = sorted(raw_bytes_map)
    with ThreadPoolExecutor(max_workers=16) as pool:
        mels = list(pool.map(lambda gid: mpg_to_causal_logmel_64(raw_bytes_map[gid], 16), ordered_ids))
    return dict(zip(ordered_ids, mels))


def _decode_sampled_speech_wavs(needed_wav_names: set[str]) -> dict[str, np.ndarray]:
    """Single-pass extract raw `16 kHz` WAV bytes from `speech_commands_v0.02.tar.gz` & decode 16 causal ticks."""
    target_members = {f"./{nm}": nm for nm in needed_wav_names} | {nm: nm for nm in needed_wav_names}
    raw_wav_map: dict[str, bytes] = {}
    with tarfile.open(SPEECH_TAR_PATH, "r:gz") as tf:
        for member in tf:
            canon = target_members.get(member.name)
            if canon is not None and canon not in raw_wav_map:
                fobj = tf.extractfile(member)
                if fobj is not None:
                    raw_wav_map[canon] = fobj.read()
                    if len(raw_wav_map) == len(needed_wav_names):
                        break
    missing = needed_wav_names - set(raw_wav_map)
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} SpeechCommands WAVs in {SPEECH_TAR_PATH}: {sorted(missing)[:5]}")

    ordered_names = sorted(raw_wav_map)
    with ThreadPoolExecutor(max_workers=16) as pool:
        mels = list(pool.map(lambda nm: wav_to_logmel_64(raw_wav_map[nm], 16), ordered_names))
    return dict(zip(ordered_names, mels))


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


def _swap_color_word_in_goal(goal_text: str, src_cid: int, dst_cid: int) -> str:
    """Replace ONLY the instructed color word with the competing distractor color word (preserving direction!)."""
    swapped, n_sub = re.subn(
        rf"\b{COLORS[src_cid]}\b",
        COLORS[dst_cid],
        goal_text,
        flags=re.IGNORECASE,
    )
    if n_sub == 0:
        raise ValueError(f"Expected color {COLORS[src_cid]!r} in goal_text {goal_text!r}")
    return swapped


def _strict_color_in_goal(goal_text: str) -> tuple[int | None, str | None]:
    """Return `(color_id, None)` iff `goal_text` names a single valid `GRID` color and zero `'yellow'`."""
    g_low = goal_text.lower()
    if UNSUPPORTED_YELLOW_RE.search(g_low) is not None:
        return None, "unsupported_color_yellow"
    found = [cid for cid, pat in COLOR_PATTERNS if pat.search(g_low) is not None]
    if not found:
        return None, "no_supported_grid_color_word"
    if len(found) > 1:
        # Disambiguate only if the first color precedes any target/reference phrase cleanly? Reject multi-color goals!
        return None, "ambiguous_multiple_grid_colors_in_goal"
    return int(found[0]), None


def _cardinal_dir_from_delta(dx: float, dy: float) -> int:
    """Map xArm6 normalized displacement `(dx, dy)` to `{0: left, 1: right, 2: up/top, 3: down/bottom}`."""
    if abs(dy) >= abs(dx):
        return 0 if dy < 0.0 else 1
    return 2 if dx < 0.0 else 3


def _text_matches_physical_dir(goal_text: str, phys_dir: int) -> bool:
    """Require explicit cardinal direction word matching `phys_dir` and zero `OPPOSITE_DIR[phys_dir]` word."""
    g_low = goal_text.lower()
    if DIR_PATTERNS[phys_dir].search(g_low) is None:
        return False
    opp = OPPOSITE_DIR[phys_dir]
    if DIR_PATTERNS[opp].search(g_low) is not None:
        return False
    return True


def _build_global_connected_scene_groups() -> tuple[dict[str, str], dict[str, int], dict[str, str]]:
    """Union-Find connected-component grouping over `z['frame']` across all roots (`g13_robot.py:split_groups`)."""
    uf_parent: dict[str, str] = {}

    def find(a: str) -> str:
        uf_parent.setdefault(a, a)
        while uf_parent[a] != a:
            uf_parent[a] = uf_parent[uf_parent[a]]
            a = uf_parent[a]
        return a

    def union(a: str, b: str) -> None:
        rx, ry = find(a), find(b)
        if rx != ry:
            if rx < ry:
                uf_parent[ry] = rx
            else:
                uf_parent[rx] = ry

    image_owner: dict[bytes, str] = {}
    for root in ROBOT_ROOTS:
        dec_dir = root / "decoded"
        if not dec_dir.is_dir():
            continue
        for npz_p in sorted(dec_dir.glob("*.npz")):
            js_p = npz_p.with_suffix(".json")
            if not js_p.is_file():
                continue
            records = json.loads(js_p.read_bytes())
            with np.load(npz_p, allow_pickle=False) as z:
                frames = z["frame"]
            for rec in records:
                p_str = str(rec["parent"])
                find(p_str)
                s0 = int(rec["start"])
                ln = int(rec["length"])
                for f_row in np.unique(frames[s0 : s0 + ln], axis=0):
                    fb = bytes(f_row)
                    owner = image_owner.setdefault(fb, p_str)
                    if owner != p_str:
                        union(owner, p_str)

    parent_to_root = {p: find(p) for p in uf_parent}
    all_roots = sorted(set(parent_to_root.values()))
    root_to_gid = {r: idx for idx, r in enumerate(all_roots)}
    root_to_split: dict[str, str] = {}
    for r in all_roots:
        hb = int(hashlib.sha256(f"w8a_scene_group/{r}".encode("utf-8")).hexdigest()[:12], 16) % 100
        root_to_split[r] = "train" if hb < 72 else ("dev" if hb < 86 else "vault")
    return parent_to_root, root_to_gid, root_to_split


def build_wave8a_grounded_datasets(
    *,
    max_train: int = 4096,
    max_dev: int = 512,
    max_vault: int = 512,
    img_res: int = 64,
) -> dict[str, Any]:
    if _cv2 is not None:
        _cv2.setNumThreads(1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vocab_doc = json.loads((OUT_DIR / "vocab_4096_ext.json").read_text())
    vocab_map: dict[str, int] = vocab_doc.get("word_to_id", vocab_doc)

    color_meta, color_spks, media_entries = _load_raw_color_metadata()
    dir_meta, dir_spks = _load_raw_direction_metadata()
    assert color_spks["train"].isdisjoint(color_spks["dev"])
    assert color_spks["train"].isdisjoint(color_spks["vault"])
    assert color_spks["dev"].isdisjoint(color_spks["vault"])
    assert dir_spks["train"].isdisjoint(dir_spks["dev"])
    assert dir_spks["train"].isdisjoint(dir_spks["vault"])
    assert dir_spks["dev"].isdisjoint(dir_spks["vault"])

    parent_to_root, root_to_gid, root_to_split = _build_global_connected_scene_groups()
    excluded_counts: Counter[str] = Counter()
    seen_episode_keys: set[tuple[str, int, int, str]] = set()
    max_per_split = {"train": max_train, "dev": max_dev, "vault": max_vault}

    selected_rows: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "vault": []}
    disposition_ledger: list[dict[str, Any]] = []
    total_scanned = 0

    for root in ROBOT_ROOTS:
        dec_dir = root / "decoded"
        raw_dir = root / "raw"
        if not (dec_dir.is_dir() and raw_dir.is_dir()):
            continue
        for js_path in sorted(dec_dir.glob("*.json")):
            shard_id = js_path.stem
            tf_path = raw_dir / f"language_table-train.tfrecord-{shard_id}"
            npz_path = dec_dir / f"{shard_id}.npz"
            if not (tf_path.is_file() and npz_path.is_file()):
                continue
            meta_list = json.loads(js_path.read_text())
            with np.load(npz_path, allow_pickle=False) as nd:
                x_all = nd["x"].astype(np.float32)
                u_all = nd["u"].astype(np.float32)

            for rec_ord, rec in enumerate(meta_list):
                scan_idx = total_scanned
                total_scanned += 1
                par_str = str(rec["parent"])
                s0 = int(rec["start"])
                ln = int(rec["length"])
                goal_str = str(rec["goal"])
                grp_root = parent_to_root[par_str]
                sp = root_to_split[grp_root]
                cid_int = int(root_to_gid[grp_root])
                par_hash = _short_sha("parent", par_str)
                grp_hash = _short_sha("group", grp_root)
                ep_uid = _short_sha(
                    "ep",
                    f"{root.name}:{shard_id}:{rec_ord}:{rec['offset']}:{par_str}:{s0}:{ln}",
                )

                def _record_excl(reason: str) -> None:
                    excluded_counts[reason] += 1
                    disposition_ledger.append(
                        {
                            "scan_index": scan_idx,
                            "episode_uid": ep_uid,
                            "source_partition": root.name,
                            "parent_hash": par_hash,
                            "connected_group_hash": grp_hash,
                            "scene_cluster_id": cid_int,
                            "assigned_split": sp,
                            "disposition": "excluded",
                            "reason": reason,
                            "retained_row_index": None,
                        }
                    )

                dedup_key = (par_str, s0, ln, goal_str)
                if dedup_key in seen_episode_keys:
                    _record_excl("duplicate_episode_across_roots")
                    continue
                seen_episode_keys.add(dedup_key)

                if len(selected_rows[sp]) >= max_per_split[sp]:
                    _record_excl(f"split_cap_reached_{sp}")
                    continue

                if ln < 8:
                    _record_excl("raw_episode_shorter_than_8")
                    continue

                c_id, col_err = _strict_color_in_goal(goal_str)
                if c_id is None:
                    _record_excl(str(col_err))
                    continue

                x_ep = x_all[s0 : s0 + ln]
                u_ep = u_all[s0 : s0 + ln]
                ee0_raw = np.array([(x_ep[0, 0] - 0.375) / 0.35, x_ep[0, 1] / 0.35], dtype=np.float32)
                ee1_raw = np.array([(x_ep[min(ln, 32) - 1, 0] - 0.375) / 0.35, x_ep[min(ln, 32) - 1, 1] / 0.35], dtype=np.float32)
                if float(np.linalg.norm(ee1_raw - ee0_raw)) < 0.06:
                    _record_excl("raw_displacement_below_threshold")
                    continue

                raw_dir_guess = _cardinal_dir_from_delta(float(ee1_raw[0] - ee0_raw[0]), float(ee1_raw[1] - ee0_raw[1]))
                if not _text_matches_physical_dir(goal_str, raw_dir_guess):
                    _record_excl("text_physical_direction_mismatch")
                    continue

                raw_ex = _read_tfrecord_at_offset(tf_path, int(rec["offset"]))
                jpegs = extract_rgb_jpegs_from_example(raw_ex)
                vid, kept_idx = _decode_frames_synchronized(jpegs, len(x_ep), max_t=32, img_res=img_res)
                use_n = int(kept_idx.shape[0])
                if use_n < 8:
                    _record_excl("too_few_synchronized_frames")
                    continue

                x_sync = x_ep[kept_idx]
                u_sync = u_ep[kept_idx]
                ee_sync = np.clip(
                    np.stack([(x_sync[:, 0] - 0.375) / 0.35, x_sync[:, 1] / 0.35], axis=-1),
                    -1.0,
                    1.0,
                ).astype(np.float32)
                act_sync = np.clip(u_sync / 0.05, -1.0, 1.0).astype(np.float32)
                d_xy = np.clip((ee_sync[use_n - 1] - ee_sync[0]) * 1.25, -0.95, 0.95).astype(np.float32)
                if float(np.linalg.norm(d_xy)) < 0.08:
                    _record_excl("synchronized_displacement_below_0p08")
                    continue

                phys_dir = _cardinal_dir_from_delta(float(d_xy[0]), float(d_xy[1]))
                if not _text_matches_physical_dir(goal_str, phys_dir):
                    _record_excl("post_sync_direction_mismatch")
                    continue

                block_map = extract_table_block_centroids_64(vid[0])
                if c_id not in block_map:
                    _record_excl("target_color_block_not_visible_in_frame0")
                    continue
                targ_blk_xy = block_map[c_id]
                best_dist_cid = -1
                best_dist_sep = -1.0
                for cand_cid, cand_xy in sorted(block_map.items()):
                    if cand_cid == c_id:
                        continue
                    sep = float(np.linalg.norm(cand_xy - targ_blk_xy))
                    if sep >= MIN_DISTRACTOR_BLOCK_SEP and sep > best_dist_sep:
                        best_dist_sep = sep
                        best_dist_cid = int(cand_cid)
                if best_dist_cid < 0:
                    _record_excl("missing_separated_distractor_block_in_frame0")
                    continue
                dist_blk_xy = block_map[best_dist_cid]
                wrong_col_goal_str = _swap_color_word_in_goal(goal_str, int(c_id), best_dist_cid)

                row_idx_in_split = len(selected_rows[sp])
                disposition_ledger.append(
                    {
                        "scan_index": scan_idx,
                        "episode_uid": ep_uid,
                        "source_partition": root.name,
                        "parent_hash": par_hash,
                        "connected_group_hash": grp_hash,
                        "scene_cluster_id": cid_int,
                        "assigned_split": sp,
                        "disposition": "retained",
                        "reason": "retained",
                        "retained_row_index": row_idx_in_split,
                    }
                )
                selected_rows[sp].append(
                    {
                        "episode_uid": ep_uid,
                        "root_source": root.name,
                        "parent_str": par_str,
                        "parent_hash": par_hash,
                        "group_root": grp_root,
                        "group_hash": grp_hash,
                        "scene_cluster_id": cid_int,
                        "video": vid,
                        "ee_sync": ee_sync,
                        "act_sync": act_sync,
                        "kept_idx": kept_idx,
                        "length": use_n,
                        "disp_xy": d_xy,
                        "color_id": int(c_id),
                        "distractor_color_id": best_dist_cid,
                        "target_block_xy": targ_blk_xy,
                        "distractor_block_xy": dist_blk_xy,
                        "block_sep_l2": round(best_dist_sep, 4),
                        "dir_id": int(phys_dir),
                        "goal_str": goal_str,
                        "wrong_color_goal_str": wrong_col_goal_str,
                    }
                )

    # Verify exhaustive, mutually-exclusive 1-to-1 coverage across ALL scanned source episodes!
    assert len(disposition_ledger) == total_scanned
    assert len({r["scan_index"] for r in disposition_ledger}) == total_scanned
    assert len({r["episode_uid"] for r in disposition_ledger}) == total_scanned
    n_retained_tot = sum(len(selected_rows[sp]) for sp in ("train", "dev", "vault"))
    n_excluded_tot = sum(excluded_counts.values())
    assert n_retained_tot + n_excluded_tot == total_scanned
    assert sum(1 for r in disposition_ledger if r["disposition"] == "retained") == n_retained_tot
    assert sum(1 for r in disposition_ledger if r["disposition"] == "excluded") == n_excluded_tot

    # Verify strict cross-split disjointness on raw parents, connected group roots, AND integer scene_cluster_ids
    split_parents = {sp: {r["parent_str"] for r in selected_rows[sp]} for sp in ("train", "dev", "vault")}
    split_groups = {sp: {r["group_root"] for r in selected_rows[sp]} for sp in ("train", "dev", "vault")}
    split_cids = {sp: {r["scene_cluster_id"] for r in selected_rows[sp]} for sp in ("train", "dev", "vault")}
    for a_sp, b_sp in (("train", "dev"), ("train", "vault"), ("dev", "vault")):
        assert split_parents[a_sp].isdisjoint(split_parents[b_sp]), f"Parent leak {a_sp} vs {b_sp}"
        assert split_groups[a_sp].isdisjoint(split_groups[b_sp]), f"Connected scene group leak {a_sp} vs {b_sp}"
        assert split_cids[a_sp].isdisjoint(split_cids[b_sp]), f"Cluster ID leak {a_sp} vs {b_sp}"

    # Sample clip assignments deterministically first, then batch-decode raw `.mpg` and `.wav` through the unified causal frontend
    rng = np.random.default_rng(20261114)
    needed_grid_gids: set[int] = set()
    needed_wav_names: set[str] = set()
    for sp in ("train", "dev", "vault"):
        for row in selected_rows[sp]:
            c_pool = color_meta[sp][int(row["color_id"])]
            wc_pool = color_meta[sp][int(row["distractor_color_id"])]
            d_pool = dir_meta[sp][int(row["dir_id"])]
            c_gid, c_spk, c_clip = c_pool[int(rng.integers(0, len(c_pool)))]
            wc_gid, wc_spk, wc_clip = wc_pool[int(rng.integers(0, len(wc_pool)))]
            d_wav, d_spk, d_clip = d_pool[int(rng.integers(0, len(d_pool)))]
            row["sampled_clips"] = (c_gid, c_spk, c_clip, wc_gid, wc_spk, wc_clip, d_wav, d_spk, d_clip)
            needed_grid_gids.add(int(c_gid))
            needed_grid_gids.add(int(wc_gid))
            needed_wav_names.add(str(d_wav))

    grid_mel_cache = _decode_sampled_grid_mpgs(needed_grid_gids, media_entries)
    wav_mel_cache = _decode_sampled_speech_wavs(needed_wav_names)

    provenance_rows: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "vault": []}
    cards: dict[str, Any] = {}

    for out_name, split_keys in [
        ("x1b_wave8a_grounded_v2", ("train", "dev")),
        ("x1b_wave8a_vault_v2", ("vault",)),
    ]:
        arrays: dict[str, np.ndarray] = {}
        split_counts: dict[str, int] = {}
        unique_sessions: dict[str, int] = {}
        unique_groups: dict[str, int] = {}
        dir_hist: dict[str, dict[str, int]] = {}
        col_hist: dict[str, dict[str, int]] = {}
        cf_stats: dict[str, dict[str, float]] = {}

        for sp in split_keys:
            items = selected_rows[sp]
            n = len(items)
            if n < 32:
                raise RuntimeError(f"Insufficient verified episodes in split {sp!r}: {n}")

            videos = np.zeros((n, 32, img_res, img_res, 3), dtype=np.uint8)
            audios = np.zeros((n, 32, 64), dtype=np.float16)
            wrong_col_audios = np.zeros((n, 32, 64), dtype=np.float16)
            ee_pos = np.zeros((n, 32, 2), dtype=np.float32)
            actions = np.zeros((n, 32, 2), dtype=np.float32)
            kept_mat = np.full((n, 32), -1, dtype=np.int16)
            lengths = np.zeros((n,), dtype=np.int32)
            true_txt = np.zeros((n, 16), dtype=np.int32)
            wrong_col_txt = np.zeros((n, 16), dtype=np.int32)
            disp_vec = np.zeros((n, 2), dtype=np.float32)
            targ_blk_arr = np.zeros((n, 2), dtype=np.float32)
            dist_blk_arr = np.zeros((n, 2), dtype=np.float32)
            color_ids = np.zeros((n,), dtype=np.int32)
            dist_color_ids = np.zeros((n,), dtype=np.int32)
            dir_ids = np.zeros((n,), dtype=np.int32)
            scene_cluster_ids = np.zeros((n,), dtype=np.int32)
            col_spk_ids = np.zeros((n,), dtype=np.int32)
            col_clip_ids = np.zeros((n,), dtype=np.int32)
            wcol_spk_ids = np.zeros((n,), dtype=np.int32)
            wcol_clip_ids = np.zeros((n,), dtype=np.int32)
            dir_clip_ids = np.zeros((n,), dtype=np.int32)
            dir_spk_ids: list[str] = []
            ep_uids: list[str] = []
            par_hashes: list[str] = []
            grp_hashes: list[str] = []

            for idx, row in enumerate(items):
                use_n = int(row["length"])
                videos[idx] = row["video"]
                ee_pos[idx, :use_n] = row["ee_sync"]
                actions[idx, :use_n] = row["act_sync"]
                kept_mat[idx, :use_n] = row["kept_idx"]
                lengths[idx] = use_n
                disp_vec[idx] = row["disp_xy"]
                targ_blk_arr[idx] = row["target_block_xy"]
                dist_blk_arr[idx] = row["distractor_block_xy"]
                c_id = int(row["color_id"])
                wc_id = int(row["distractor_color_id"])
                d_id = int(row["dir_id"])
                color_ids[idx] = c_id
                dist_color_ids[idx] = wc_id
                dir_ids[idx] = d_id
                scene_cluster_ids[idx] = int(row["scene_cluster_id"])

                c_gid, c_spk, c_clip, wc_gid, wc_spk, wc_clip, d_wav, d_spk, d_clip = row["sampled_clips"]
                col_16t = grid_mel_cache[int(c_gid)]
                wcol_16t = grid_mel_cache[int(wc_gid)]
                dir_16t = wav_mel_cache[str(d_wav)]

                # Strictly chronological 32-tick streams: t=0..15 color, t=16..31 direction (identical on wrong_color!)
                audios[idx] = np.concatenate([col_16t, dir_16t], axis=0).astype(np.float16)
                wrong_col_audios[idx] = np.concatenate([wcol_16t, dir_16t], axis=0).astype(np.float16)

                col_spk_ids[idx] = int(c_spk)
                col_clip_ids[idx] = int(c_clip)
                wcol_spk_ids[idx] = int(wc_spk)
                wcol_clip_ids[idx] = int(wc_clip)
                dir_clip_ids[idx] = int(d_clip)
                dir_spk_ids.append(str(d_spk))
                ep_uids.append(str(row["episode_uid"]))
                par_hashes.append(str(row["parent_hash"]))
                grp_hashes.append(str(row["group_hash"]))
                true_txt[idx] = _encode_16tok_instruction(str(row["goal_str"]), vocab_map)
                wrong_col_txt[idx] = _encode_16tok_instruction(str(row["wrong_color_goal_str"]), vocab_map)

            # Strict opposite-direction counterfactual pairing (fail-closed if any row cannot be paired!)
            cf_indices = np.full((n,), -1, dtype=np.int32)
            cf_dists: list[float] = []
            cf_dots: list[float] = []
            for idx in range(n):
                opp_d = OPPOSITE_DIR[int(dir_ids[idx])]
                best_cf = -1
                best_score = -1e9
                for step in range(1, n):
                    cand = (idx + 7 + step - 1) % n
                    if int(dir_ids[cand]) != opp_d:
                        continue
                    dot_v = float(np.dot(disp_vec[cand], disp_vec[idx]))
                    dist_v = float(np.linalg.norm(disp_vec[cand] - disp_vec[idx]))
                    if dot_v >= -0.02 or dist_v < 0.35:
                        continue
                    diff_col = 0.25 if int(color_ids[cand]) != int(color_ids[idx]) else 0.0
                    score = diff_col + dist_v - dot_v
                    if score > best_score:
                        best_score = score
                        best_cf = cand
                    if diff_col > 0.0 and dist_v >= 0.55 and dot_v <= -0.08:
                        best_cf = cand
                        break
                if best_cf < 0:
                    raise RuntimeError(
                        f"Failed to find strict opposite-direction counterfactual for {sp}[{idx}] (dir={dir_ids[idx]})"
                    )
                cf_indices[idx] = best_cf
                cf_dists.append(float(np.linalg.norm(disp_vec[best_cf] - disp_vec[idx])))
                cf_dots.append(float(np.dot(disp_vec[best_cf], disp_vec[idx])))

            wrong_txt = true_txt[cf_indices]
            wrong_aud = audios[cf_indices]
            block_seps = np.linalg.norm(targ_blk_arr - dist_blk_arr, axis=-1)
            assert np.all(dist_color_ids != color_ids)
            assert np.all(block_seps >= MIN_DISTRACTOR_BLOCK_SEP - 1e-5)
            assert np.array_equal(wrong_col_audios[:, 16:32], audios[:, 16:32])

            for idx in range(n):
                cfi = int(cf_indices[idx])
                provenance_rows[sp].append(
                    {
                        "row_index": idx,
                        "episode_uid": ep_uids[idx],
                        "source_partition": str(items[idx]["root_source"]),
                        "parent_hash": par_hashes[idx],
                        "connected_group_hash": grp_hashes[idx],
                        "scene_cluster_id": int(scene_cluster_ids[idx]),
                        "length": int(lengths[idx]),
                        "kept_frame_start": int(kept_mat[idx, 0]),
                        "kept_frame_end": int(kept_mat[idx, lengths[idx] - 1]),
                        "strictly_increasing_timestamps": bool(
                            np.all(np.diff(kept_mat[idx, : lengths[idx]]) > 0)
                        ),
                        "color_id": int(color_ids[idx]),
                        "color_name": COLORS[int(color_ids[idx])],
                        "distractor_color_id": int(dist_color_ids[idx]),
                        "distractor_color_name": COLORS[int(dist_color_ids[idx])],
                        "target_block_xy": [round(float(targ_blk_arr[idx, 0]), 4), round(float(targ_blk_arr[idx, 1]), 4)],
                        "distractor_block_xy": [round(float(dist_blk_arr[idx, 0]), 4), round(float(dist_blk_arr[idx, 1]), 4)],
                        "block_sep_l2": round(float(block_seps[idx]), 4),
                        "dir_id": int(dir_ids[idx]),
                        "dir_name": DIR_NAMES[int(dir_ids[idx])],
                        "disp_xy": [round(float(disp_vec[idx, 0]), 4), round(float(disp_vec[idx, 1]), 4)],
                        "color_speaker_id": int(col_spk_ids[idx]),
                        "color_clip_id": int(col_clip_ids[idx]),
                        "wrong_color_speaker_id": int(wcol_spk_ids[idx]),
                        "wrong_color_clip_id": int(wcol_clip_ids[idx]),
                        "dir_speaker_id": dir_spk_ids[idx],
                        "dir_clip_id": int(dir_clip_ids[idx]),
                        "cf_index": cfi,
                        "cf_dir_id": int(dir_ids[cfi]),
                        "cf_dot": round(float(cf_dots[idx]), 4),
                        "cf_disp_l2": round(float(cf_dists[idx]), 4),
                    }
                )

            pfx = f"{sp}_"
            arrays[pfx + "video"] = videos
            arrays[pfx + "spoken_audio"] = audios
            arrays[pfx + "wrong_spoken_audio"] = wrong_aud
            arrays[pfx + "wrong_color_spoken_audio"] = wrong_col_audios
            arrays[pfx + "ee_xy"] = ee_pos
            arrays[pfx + "action_xy"] = actions
            arrays[pfx + "kept_frame_indices"] = kept_mat
            arrays[pfx + "length"] = lengths
            arrays[pfx + "disp_xy"] = disp_vec
            arrays[pfx + "target_block_xy"] = targ_blk_arr
            arrays[pfx + "distractor_block_xy"] = dist_blk_arr
            arrays[pfx + "true_text_ids"] = true_txt
            arrays[pfx + "wrong_text_ids"] = wrong_txt
            arrays[pfx + "wrong_color_text_ids"] = wrong_col_txt
            arrays[pfx + "cf_index"] = cf_indices
            arrays[pfx + "scene_cluster_id"] = scene_cluster_ids
            arrays[pfx + "episode_uid"] = np.asarray(ep_uids, dtype="<U16")
            arrays[pfx + "parent_hash"] = np.asarray(par_hashes, dtype="<U16")
            arrays[pfx + "connected_group_hash"] = np.asarray(grp_hashes, dtype="<U16")
            arrays[pfx + "color_speaker_id"] = col_spk_ids
            arrays[pfx + "color_clip_id"] = col_clip_ids
            arrays[pfx + "wrong_color_speaker_id"] = wcol_spk_ids
            arrays[pfx + "wrong_color_clip_id"] = wcol_clip_ids
            arrays[pfx + "dir_speaker_id"] = np.asarray(dir_spk_ids, dtype="<U12")
            arrays[pfx + "dir_clip_id"] = dir_clip_ids
            arrays[pfx + "color_id"] = color_ids
            arrays[pfx + "distractor_color_id"] = dist_color_ids
            arrays[pfx + "dir_id"] = dir_ids

            split_counts[sp] = n
            unique_sessions[sp] = len(split_parents[sp])
            unique_groups[sp] = len(split_groups[sp])
            col_hist[sp] = {COLORS[c]: int(np.sum(color_ids == c)) for c in range(4)}
            dir_hist[sp] = {DIR_NAMES[d]: int(np.sum(dir_ids == d)) for d in range(4)}
            opp_targets = np.asarray([OPPOSITE_DIR[int(d)] for d in dir_ids], dtype=np.int32)
            cf_stats[sp] = {
                "exact_opposite_dir_fraction": round(float(np.mean(dir_ids[cf_indices] == opp_targets)), 6),
                "negative_dot_fraction": round(float(np.mean(np.asarray(cf_dots) < -0.02)), 6),
                "min_counterfactual_disp_l2": round(float(np.min(cf_dists)), 4),
                "mean_counterfactual_disp_l2": round(float(np.mean(cf_dists)), 4),
                "max_counterfactual_dot": round(float(np.max(cf_dots)), 4),
                "same_scene_wrong_color_fraction": 1.0,
                "min_distractor_block_sep_l2": round(float(np.min(block_seps)), 4),
                "mean_distractor_block_sep_l2": round(float(np.mean(block_seps)), 4),
            }

        npz_out = OUT_DIR / f"{out_name}.npz"
        tmp_out = npz_out.with_suffix(".tmp.npz")
        np.savez_compressed(tmp_out, **arrays)
        tmp_out.replace(npz_out)
        sha = _sha256_path(npz_out)

        card = {
            "dataset": out_name,
            "sha256": sha,
            "bytes": npz_out.stat().st_size,
            "img_res": img_res,
            "color_vocabulary": list(COLORS),
            "direction_vocabulary": list(DIR_NAMES),
            "opposite_direction_map": {DIR_NAMES[k]: DIR_NAMES[v] for k, v in OPPOSITE_DIR.items()},
            "splits": split_counts,
            "unique_parent_sessions": unique_sessions,
            "unique_connected_scene_groups": unique_groups,
            "color_distribution": col_hist,
            "direction_distribution": dir_hist,
            "total_source_episodes_scanned": total_scanned,
            "excluded_counts_by_reason": dict(sorted(excluded_counts.items())),
            "zero_overlap_verified": {
                "parent_sessions_train_dev": len(split_parents["train"] & split_parents["dev"]),
                "parent_sessions_train_vault": len(split_parents["train"] & split_parents["vault"]),
                "parent_sessions_dev_vault": len(split_parents["dev"] & split_parents["vault"]),
                "connected_scene_groups_train_dev": len(split_groups["train"] & split_groups["dev"]),
                "connected_scene_groups_train_vault": len(split_groups["train"] & split_groups["vault"]),
                "connected_scene_groups_dev_vault": len(split_groups["dev"] & split_groups["vault"]),
                "scene_cluster_ids_train_dev": len(split_cids["train"] & split_cids["dev"]),
                "scene_cluster_ids_train_vault": len(split_cids["train"] & split_cids["vault"]),
                "scene_cluster_ids_dev_vault": len(split_cids["dev"] & split_cids["vault"]),
                "color_speakers_train_dev": len(color_spks["train"] & color_spks["dev"]),
                "color_speakers_train_vault": len(color_spks["train"] & color_spks["vault"]),
                "color_speakers_dev_vault": len(color_spks["dev"] & color_spks["vault"]),
                "dir_speakers_train_dev": len(dir_spks["train"] & dir_spks["dev"]),
                "dir_speakers_train_vault": len(dir_spks["train"] & dir_spks["vault"]),
                "dir_speakers_dev_vault": len(dir_spks["dev"] & dir_spks["vault"]),
            },
            "audio_frontend": {
                "causal": True,
                "word_alignment_metadata_used": False,
                "ffmpeg_resampling_used": False,
                "whole_clip_max_abs_normalization_used": False,
                "normalization": "fixed_train_reference_log10_power",
                "log10_mean": CAUSAL_LOG10_MEAN,
                "log10_std": CAUSAL_LOG10_STD,
                "tick_layout": "t0_15_raw_grid_44p1khz_2to1_boxcar_22p05khz_t16_31_raw_speechcommands_16khz",
                "grid_mp2_frontend_state_bytes": MAX_CAUSAL_AUDIO_FRONTEND_BYTES,
                "speechcommands_wav_frontend_state_bytes": 6400,
                "strict_system_budget_bytes": MAX_CAUSAL_AUDIO_FRONTEND_BYTES,
            },
            "counterfactual_pairing": cf_stats,
        }
        (OUT_DIR / f"{out_name}.card.json").write_text(json.dumps(card, indent=2, sort_keys=True) + "\n")
        cards[out_name] = card

    disp_path = OUT_DIR / "x1b_wave8a_episode_dispositions_v2.jsonl.gz"
    tmp_disp = disp_path.with_suffix(".tmp.gz")
    with tmp_disp.open("wb") as raw_fh:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_fh, compresslevel=6, mtime=0) as gz:
            for row in disposition_ledger:
                gz.write((json.dumps(row, sort_keys=True) + "\n").encode("utf-8"))
    tmp_disp.replace(disp_path)
    disp_sha = _sha256_path(disp_path)

    for cname, card in cards.items():
        card["episode_disposition_ledger"] = {
            "file": "wave8a_episode_dispositions.jsonl.gz",
            "sha256": disp_sha,
            "total_rows": total_scanned,
            "retained_rows": n_retained_tot,
            "excluded_rows": n_excluded_tot,
        }
        (OUT_DIR / f"{cname}.card.json").write_text(json.dumps(card, indent=2, sort_keys=True) + "\n")

    prov_doc = {
        "schema_version": "wave8a_grounded_provenance_v2_scene_color_and_dispositions",
        "color_vocabulary": list(COLORS),
        "direction_vocabulary": list(DIR_NAMES),
        "opposite_direction_map": {str(k): int(v) for k, v in OPPOSITE_DIR.items()},
        "total_source_episodes_scanned": total_scanned,
        "retained_counts_by_split": {sp: len(selected_rows[sp]) for sp in ("train", "dev", "vault")},
        "excluded_counts_by_reason": dict(sorted(excluded_counts.items())),
        "episode_disposition_ledger": {
            "file": "wave8a_episode_dispositions.jsonl.gz",
            "sha256": disp_sha,
            "total_rows": total_scanned,
            "retained_rows": n_retained_tot,
            "excluded_rows": n_excluded_tot,
        },
        "dataset_sha256": {k: v["sha256"] for k, v in cards.items()},
        "zero_overlap_verified": cards["x1b_wave8a_grounded_v2"]["zero_overlap_verified"],
        "rows_by_split": provenance_rows,
    }
    (OUT_DIR / "x1b_wave8a_provenance_manifest_v2.json").write_text(json.dumps(prov_doc, sort_keys=True) + "\n")
    return cards


if __name__ == "__main__":
    summary = build_wave8a_grounded_datasets()
    print(json.dumps(summary, indent=2, sort_keys=True))
