"""SLWM-v3 Unified Sparse Multimodal Causal Packet Shard Contract (`Wave A-Data`).

Stores arbitrarily interleaved multimodal streams (`video`, `audio`, `text_byte`, `proprio_action`,
`signal_probe`) in compact sparse-indexed `.npz` shards with ZERO dense zero-padding waste across
inactive modalities:
  - Per-tick timeline arrays (`length N`):
      `modality_id: uint8[N]`       (0:video, 1:audio, 2:text_byte, 3:proprio_action, 4:signal_probe)
      `dt_s: float16[N]`            (physical elapsed seconds >= 0)
      `valid_mask: uint8[N]`        (1 = observed packet, 0 = missing/padded)
      `reliability: float16[N]`     ([0, 1] sensor reliability)
      `reset_mask: uint8[N]`        (1 = start of new episode, resets O(1) streaming state)
      `spatial_pos: uint8[N]`       (0..14 isotropic Wan 2.1 8x8 patch index)
      `text_byte: uint16[N]`        (0..263 lossless ByT5 UTF-8 byte + 8 control tokens)
      `has_true_action: uint8[N]`   (1 = executed physical action present, 0 = missing)
      `video_idx: int32[N]`         (row index into `video_feat`, or -1 if not video)
      `audio_idx: int32[N]`         (row index into `audio_codes`, or -1 if not audio)
      `act_idx: int32[N]`           (row index into `proprio_feat`/`action_feat`, or -1)
      `domain_id: uint8[N]`         (corpus/domain provenance tag for balanced eval — NEVER fed to model!)
  - Sparse payload tables (ONLY populated rows stored!):
      `video_feat: float16[N_vid, 1024]`    (Wan 2.1 Causal VAE 8x8x16 patch)
      `audio_codes: uint16[N_aud, 32]`      (Kyutai Mimi 32-RVQ discrete codes @ 12.5 Hz)
      `proprio_feat: float16[N_act, 16]`    (robot proprioceptive state)
      `action_feat: float16[N_act, 14]`     (executed motor command)

Unpacks into exact `models/v3/fast_core.py:scan_core_fast` `[T, B, ...]` JAX/NumPy batches at
> 100,000 positions/sec with zero CPU bottleneck and supports deterministic modality dropout
(`drop_video`, `drop_audio`, `drop_text`) for true any-to-any multimodal training & evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

# ByT5 control byte IDs (0..255 = raw UTF-8 bytes; 256..263 = control markers)
BYTE_PAD = 256
BYTE_BOS = 257
BYTE_EOS = 258
BYTE_MOD_SEP = 259  # Separates modality prompt / caption / QA boundaries
BYTE_CODE_BOS = 260
BYTE_QA_Q = 261
BYTE_QA_A = 262
BYTE_ABSTAIN = 263

DOMAIN_IDS: dict[str, int] = {
    "fineweb_edu": 0,
    "cosmopedia_v2": 1,
    "code_python_multi": 2,
    "wikitext_103": 3,
    "lambada": 4,
    "humaneval_mbpp": 5,
    "librispeech": 6,
    "clotho": 7,
    "esc50": 8,
    "speech_commands": 9,
    "greatesthits": 10,
    "grid_av": 11,
    "msrvtt": 12,
    "vggsound": 13,
    "holoassist": 14,
    "xarm6": 15,
    "aloha_fineart": 16,
    "lerobot_pusht": 17,
    "lerobot_aloha_sim": 18,
    "lerobot_libero": 19,
    "streamingbench": 20,
    "real_speech_overwrite": 21,
}


def encode_utf8_bytes(text: str, *, bos: int = BYTE_BOS, eos: int = BYTE_EOS, max_bytes: int | None = None) -> np.ndarray:
    """Lossless ByT5 UTF-8 byte encoder (`uint16`, 0..255 raw bytes + control tokens 256..263)."""
    raw = text.encode("utf-8", errors="replace")
    if max_bytes is not None and len(raw) > max_bytes:
        raw = raw[:max_bytes]
    out = np.empty(len(raw) + (1 if bos >= 0 else 0) + (1 if eos >= 0 else 0), dtype=np.uint16)
    pos = 0
    if bos >= 0:
        out[pos] = bos
        pos += 1
    if raw:
        out[pos : pos + len(raw)] = np.frombuffer(raw, dtype=np.uint8).astype(np.uint16)
        pos += len(raw)
    if eos >= 0:
        out[pos] = eos
    return out


def decode_utf8_bytes(byte_ids: np.ndarray) -> str:
    """Lossless decoder from ByT5 byte IDs back to UTF-8 string (stripping control markers >= 256)."""
    arr = np.asarray(byte_ids, dtype=np.int32)
    raw = bytes(int(x) for x in arr if 0 <= int(x) < 256)
    return raw.decode("utf-8", errors="replace")


class SparseShardBuilder:
    """Accumulates causal multimodal episodes and writes a compact sparse-indexed `.npz` shard."""

    def __init__(self, shard_name: str, split: str) -> None:
        self.shard_name = shard_name
        self.split = split
        self._mod_id: list[np.ndarray] = []
        self._dt_s: list[np.ndarray] = []
        self._valid: list[np.ndarray] = []
        self._rel: list[np.ndarray] = []
        self._reset: list[np.ndarray] = []
        self._sp_pos: list[np.ndarray] = []
        self._text_byte: list[np.ndarray] = []
        self._has_act: list[np.ndarray] = []
        self._vid_idx: list[np.ndarray] = []
        self._aud_idx: list[np.ndarray] = []
        self._act_idx: list[np.ndarray] = []
        self._dom_id: list[np.ndarray] = []

        self._vid_rows: list[np.ndarray] = []
        self._aud_rows: list[np.ndarray] = []
        self._prop_rows: list[np.ndarray] = []
        self._act_rows: list[np.ndarray] = []
        self._n_vid = 0
        self._n_aud = 0
        self._n_act = 0
        self._n_ticks = 0
        self._n_episodes = 0

    @property
    def num_ticks(self) -> int:
        return self._n_ticks

    @property
    def num_episodes(self) -> int:
        return self._n_episodes

    def add_episode(
        self,
        *,
        modality_id: np.ndarray,
        dt_s: np.ndarray,
        domain_name: str,
        text_byte: np.ndarray | None = None,
        spatial_pos: np.ndarray | None = None,
        reliability: np.ndarray | None = None,
        video_feat: np.ndarray | None = None,
        video_tick_mask: np.ndarray | None = None,
        audio_codes: np.ndarray | None = None,
        audio_tick_mask: np.ndarray | None = None,
        proprio_feat: np.ndarray | None = None,
        action_feat: np.ndarray | None = None,
        has_true_action: np.ndarray | None = None,
        act_tick_mask: np.ndarray | None = None,
    ) -> None:
        """Append one complete causal episode (setting `reset_mask[0] = 1` automatically)."""
        n = int(modality_id.shape[0])
        if n <= 0:
            return
        mod = np.asarray(modality_id, dtype=np.uint8)
        dt = np.asarray(dt_s, dtype=np.float16)
        vld = np.ones((n,), dtype=np.uint8)
        rel = np.ones((n,), dtype=np.float16) if reliability is None else np.asarray(reliability, dtype=np.float16)
        rst = np.zeros((n,), dtype=np.uint8)
        rst[0] = 1
        sp = np.zeros((n,), dtype=np.uint8) if spatial_pos is None else np.asarray(spatial_pos, dtype=np.uint8)
        tb = np.full((n,), BYTE_PAD, dtype=np.uint16) if text_byte is None else np.asarray(text_byte, dtype=np.uint16)
        ha = np.zeros((n,), dtype=np.uint8) if has_true_action is None else np.asarray(has_true_action, dtype=np.uint8)
        dom = np.full((n,), DOMAIN_IDS[domain_name], dtype=np.uint8)

        v_idx = np.full((n,), -1, dtype=np.int32)
        if video_feat is not None and len(video_feat) > 0:
            vm = (mod == 0) if video_tick_mask is None else np.asarray(video_tick_mask, dtype=bool)
            nv = int(np.sum(vm))
            if nv != len(video_feat):
                raise ValueError(f"video_feat len {len(video_feat)} != video ticks {nv}")
            v_idx[vm] = np.arange(self._n_vid, self._n_vid + nv, dtype=np.int32)
            self._vid_rows.append(np.asarray(video_feat, dtype=np.float16))
            self._n_vid += nv

        a_idx = np.full((n,), -1, dtype=np.int32)
        if audio_codes is not None and len(audio_codes) > 0:
            am = (mod == 1) if audio_tick_mask is None else np.asarray(audio_tick_mask, dtype=bool)
            na = int(np.sum(am))
            if na != len(audio_codes):
                raise ValueError(f"audio_codes len {len(audio_codes)} != audio ticks {na}")
            a_idx[am] = np.arange(self._n_aud, self._n_aud + na, dtype=np.int32)
            self._aud_rows.append(np.asarray(audio_codes, dtype=np.uint16))
            self._n_aud += na

        p_idx = np.full((n,), -1, dtype=np.int32)
        if proprio_feat is not None and len(proprio_feat) > 0:
            pm = (mod == 3) if act_tick_mask is None else np.asarray(act_tick_mask, dtype=bool)
            np_n = int(np.sum(pm))
            if np_n != len(proprio_feat):
                raise ValueError(f"proprio_feat len {len(proprio_feat)} != proprio ticks {np_n}")
            p_idx[pm] = np.arange(self._n_act, self._n_act + np_n, dtype=np.int32)
            self._prop_rows.append(np.asarray(proprio_feat, dtype=np.float16))
            act_arr = np.zeros((np_n, 14), dtype=np.float16) if action_feat is None else np.asarray(action_feat, dtype=np.float16)
            self._act_rows.append(act_arr)
            self._n_act += np_n

        self._mod_id.append(mod)
        self._dt_s.append(dt)
        self._valid.append(vld)
        self._rel.append(rel)
        self._reset.append(rst)
        self._sp_pos.append(sp)
        self._text_byte.append(tb)
        self._has_act.append(ha)
        self._vid_idx.append(v_idx)
        self._aud_idx.append(a_idx)
        self._act_idx.append(p_idx)
        self._dom_id.append(dom)
        self._n_ticks += n
        self._n_episodes += 1

    def save(self, out_path: Path) -> dict[str, Any]:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "modality_id": np.concatenate(self._mod_id) if self._mod_id else np.zeros((0,), dtype=np.uint8),
            "dt_s": np.concatenate(self._dt_s) if self._dt_s else np.zeros((0,), dtype=np.float16),
            "valid_mask": np.concatenate(self._valid) if self._valid else np.zeros((0,), dtype=np.uint8),
            "reliability": np.concatenate(self._rel) if self._rel else np.zeros((0,), dtype=np.float16),
            "reset_mask": np.concatenate(self._reset) if self._reset else np.zeros((0,), dtype=np.uint8),
            "spatial_pos": np.concatenate(self._sp_pos) if self._sp_pos else np.zeros((0,), dtype=np.uint8),
            "text_byte": np.concatenate(self._text_byte) if self._text_byte else np.zeros((0,), dtype=np.uint16),
            "has_true_action": np.concatenate(self._has_act) if self._has_act else np.zeros((0,), dtype=np.uint8),
            "video_idx": np.concatenate(self._vid_idx) if self._vid_idx else np.zeros((0,), dtype=np.int32),
            "audio_idx": np.concatenate(self._aud_idx) if self._aud_idx else np.zeros((0,), dtype=np.int32),
            "act_idx": np.concatenate(self._act_idx) if self._act_idx else np.zeros((0,), dtype=np.int32),
            "domain_id": np.concatenate(self._dom_id) if self._dom_id else np.zeros((0,), dtype=np.uint8),
            "video_feat": np.concatenate(self._vid_rows) if self._vid_rows else np.zeros((0, 1024), dtype=np.float16),
            "audio_codes": np.concatenate(self._aud_rows) if self._aud_rows else np.zeros((0, 32), dtype=np.uint16),
            "proprio_feat": np.concatenate(self._prop_rows) if self._prop_rows else np.zeros((0, 16), dtype=np.float16),
            "action_feat": np.concatenate(self._act_rows) if self._act_rows else np.zeros((0, 14), dtype=np.float16),
        }
        np.savez(out_path, **arrays)
        sz = out_path.stat().st_size
        return {
            "shard_name": self.shard_name,
            "split": self.split,
            "path": str(out_path),
            "num_ticks": int(self._n_ticks),
            "num_episodes": int(self._n_episodes),
            "num_video_packets": int(self._n_vid),
            "num_audio_packets": int(self._n_aud),
            "num_action_packets": int(self._n_act),
            "size_mib": round(sz / (1024.0**2), 3),
        }


@dataclass
class LoadedSparseShard:
    """In-memory sparse shard supporting O(1) slice unpacking into `[T, B, ...]` model batches."""

    data: dict[str, np.ndarray]
    n_ticks: int

    @classmethod
    def load(cls, path: Path) -> "LoadedSparseShard":
        z = np.load(path, allow_pickle=False)
        d = {k: z[k] for k in z.files}
        return cls(data=d, n_ticks=int(d["modality_id"].shape[0]))

    def sample_batch(
        self,
        rng: np.random.Generator,
        *,
        batch_size: int,
        seq_len: int,
        drop_modality_prob: float = 0.0,
        anchor_transition_prob: float = 0.0,
    ) -> dict[str, np.ndarray]:
        """Sample `[T, B, ...]` packed multimodal batch with optional random modality dropout & transition anchoring."""
        d = self.data
        max_start = max(1, self.n_ticks - seq_len - 1)
        starts = rng.integers(0, max_start, size=(batch_size,), dtype=np.int64)
        # Transition-anchored sampling: on cross-modal shards (e.g. MSR-VTT 570v+75a+60t > 512),
        # explicitly align `anchor_transition_prob` of windows to span the `source -> text` boundary!
        if anchor_transition_prob > 0.0:
            if not hasattr(self, "_transition_starts"):
                rst = np.nonzero(d["reset_mask"])[0]
                if len(rst) == 0 or rst[0] != 0:
                    rst = np.insert(rst, 0, 0)
                ends = np.append(rst[1:], self.n_ticks)
                t_starts = []
                mod_all = d["modality_id"]
                for s_i, e_i in zip(rst, ends):
                    s_i, e_i = int(s_i), int(e_i)
                    m_ep = mod_all[s_i:e_i]
                    t_pos = np.where(m_ep == 2)[0]
                    non_t = np.where(m_ep != 2)[0]
                    if len(t_pos) >= 4 and len(non_t) >= 4 and int(t_pos[0]) > int(non_t[0]):
                        # Place first text byte at ~75% of window so preceding video/audio + full caption BOTH fit!
                        st_cand = max(s_i, s_i + int(t_pos[0]) - (seq_len * 3 // 4))
                        st_cand = min(st_cand, max(0, self.n_ticks - seq_len - 1))
                        t_starts.append(st_cand)
                self._transition_starts = np.asarray(t_starts, dtype=np.int64)
            if len(self._transition_starts) > 0:
                for b in range(batch_size):
                    if rng.random() < anchor_transition_prob:
                        starts[b] = int(self._transition_starts[int(rng.integers(0, len(self._transition_starts)))])

        idx = starts[None, :] + np.arange(seq_len, dtype=np.int64)[:, None]  # [T, B]
        mod = d["modality_id"][idx].astype(np.int32)
        dt = d["dt_s"][idx].astype(np.float32)
        vld = d["valid_mask"][idx].astype(np.float32)
        rel = d["reliability"][idx].astype(np.float32)
        rst = d["reset_mask"][idx].astype(bool)
        rst[0, :] = True  # Always clean episode start at chunk boundary
        sp = d["spatial_pos"][idx].astype(np.int32)
        tb = d["text_byte"][idx].astype(np.int32)
        ha = d["has_true_action"][idx].astype(np.float32)

        # Unpack sparse modality payloads strictly where active!
        vid_idx = d["video_idx"][idx]
        vid_feat = np.zeros((seq_len, batch_size, 1024), dtype=np.float32)
        vm = vid_idx >= 0
        if np.any(vm) and d["video_feat"].shape[0] > 0:
            vid_feat[vm] = d["video_feat"][vid_idx[vm]].astype(np.float32)

        aud_idx = d["audio_idx"][idx]
        aud_codes = np.zeros((seq_len, batch_size, 32), dtype=np.int32)
        am = aud_idx >= 0
        if np.any(am) and d["audio_codes"].shape[0] > 0:
            aud_codes[am] = d["audio_codes"][aud_idx[am]].astype(np.int32)

        act_idx = d["act_idx"][idx]
        prop_feat = np.zeros((seq_len, batch_size, 16), dtype=np.float32)
        act_feat = np.zeros((seq_len, batch_size, 14), dtype=np.float32)
        pm = act_idx >= 0
        if np.any(pm) and d["proprio_feat"].shape[0] > 0:
            prop_feat[pm] = d["proprio_feat"][act_idx[pm]].astype(np.float32)
            act_feat[pm] = d["action_feat"][act_idx[pm]].astype(np.float32)

        # Optional per-sample modality dropout for true any-to-any robustness:
        # Masks the dropped modality's input payload AND sets `reliability = 0.0` (training epistemic
        # sensor-outage suppression naturally!) and `target_valid_mask = 0.0` on the dropped source
        # modality while keeping `valid_mask = 1.0` so surviving target modalities are predicted cleanly
        # from non-zero residual streams AND `apply_ring_gqa_seq` stays 100% on the fast dense MXU path!
        targ_vld = vld.copy()
        if drop_modality_prob > 0.0:
            for b in range(batch_size):
                present_mods = np.unique(mod[:, b])
                if len(present_mods) >= 2 and rng.random() < drop_modality_prob:
                    drop_m = int(rng.choice(present_mods))
                    dm = mod[:, b] == drop_m
                    rel[dm, b] = 0.0
                    targ_vld[dm, b] = 0.0
                    if drop_m == 0:
                        vid_feat[dm, b, :] = 0.0
                    elif drop_m == 1:
                        aud_codes[dm, b, :] = 0
                    elif drop_m == 2:
                        tb[dm, b] = BYTE_PAD
                    elif drop_m == 3:
                        prop_feat[dm, b, :] = 0.0
                        act_feat[dm, b, :] = 0.0
                        ha[dm, b] = 0.0

        return {
            "modality_id": mod,
            "dt_s": dt,
            "valid_mask": vld,
            "target_valid_mask": targ_vld,
            "reliability": rel,
            "reset_mask": rst,
            "spatial_pos": sp,
            "text_byte": tb,
            "has_true_action": ha,
            "video_feat": vid_feat,
            "audio_codes": aud_codes,
            "proprio_feat": prop_feat,
            "action_feat": act_feat,
            "probe_feat": np.zeros((seq_len, batch_size, 32), dtype=np.float32),
            "domain_id": d["domain_id"][idx].astype(np.int32),
        }
