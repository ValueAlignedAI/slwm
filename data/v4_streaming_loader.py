"""SLWM-v4 Single-Pass (<= 1.0 Epoch) Multimodal Streaming Data Loader.

Streams pre-encoded Gate-2-certified `.npz` shards from the 10 public license-pure `valueai-org/slwm-v4-*`
repositories (or local directories for offline/unit testing) with:
  1. Exact on-the-fly certified per-patch affine `int8 -> float32` Wan 2.1 video dequantization
     (`61.44 dB` latent PSNR / `0.99957` SSIM) strictly on sliced `[T]` windows (zero full-shard RAM blowup).
  2. Strict split isolation (`train_shard_*` / `dev_shard_*` / `test_shard_*` never cross boundaries).
  3. Automatic enforcement of `reports/v4/gate2_eval_exclusion_manifest.json` (all 39 upstream cross-split
     duplicate eval episode IDs permanently blocked).
  4. Strict single-pass (`<= 1.0 epoch` per corpus) packet-budget tracking matching the exact `V4-S3` `450M`
     water-filling mixture in `reports/v4/gate2_v4_data_and_codec_receipt.json`.
  5. Bounded local disk cache rotation (deletes consumed shards automatically so TPU/GPU VM disk stays flat).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from data.v4_heartbeat_contract import SLWMv4HeartbeatSpec, UNIFIED_ACTION_DIM, UNIFIED_PROPRIO_DIM

DEFAULT_GATE2_RECEIPT = Path("reports/v4/gate2_v4_data_and_codec_receipt.json")
DEFAULT_EXCLUSION_MANIFEST = Path("reports/v4/gate2_eval_exclusion_manifest.json")

# Canonical 10 public training corpora (matching scripts/data/v4_valueai_registry.py)
V4_TRAIN_CORPUS_REPOS: dict[str, str] = {
    "unitree_g1_h1_humanoid": "valueai-org/slwm-v4-unitree-g1-h1-latents",
    "droid_1_0_1": "valueai-org/slwm-v4-droid-latents",
    "agibot_and_aloha_sim": "valueai-org/slwm-v4-agibot-aloha-latents",
    "rh20t_contact_av_force": "valueai-org/slwm-v4-rh20t-latents",
    "vggsound_full_sync": "valueai-org/slwm-v4-vggsound-latents",
    "epic_kitchens_100_long_ego_av": "valueai-org/slwm-v4-epic-kitchens-100-latents",
    "holoassist_and_greatesthits_av": "valueai-org/slwm-v4-holoassist-latents",
    "greatesthits_impact_av": "valueai-org/slwm-v4-greatesthits-latents",
    "wavcaps_and_libriheavy_speech": "valueai-org/slwm-v4-speech-audio-latents",
    "fineweb_edu_and_stack_code": "valueai-org/slwm-v4-text-code-latents",
}


def dequantize_wan21_patches_int8(
    video_q_int8: np.ndarray,
    video_xmin_fp16: np.ndarray,
    video_scale_fp16: np.ndarray,
) -> np.ndarray:
    """Exact inverse of Gate-2-certified per-patch affine `int8` Wan 2.1 quantization -> `float32[..., 1024]`."""
    if video_q_int8.shape[0] == 0:
        return np.zeros((0, video_q_int8.shape[-1]), dtype=np.float32)
    return (
        video_q_int8.astype(np.float32) * video_scale_fp16.astype(np.float32)
        + video_xmin_fp16.astype(np.float32)
    )


def load_excluded_episode_ids(manifest_path: Path = DEFAULT_EXCLUSION_MANIFEST) -> set[tuple[str, str, str]]:
    """Load permanently excluded `(corpus, split, episode_id)` tuples from Gate 2 exclusion manifest."""
    if not manifest_path.is_file():
        return set()
    data = json.loads(manifest_path.read_text())
    out: set[tuple[str, str, str]] = set()
    for row in data.get("exclusions", []):
        out.add((str(row["corpus"]), str(row["excluded_split"]), str(row["episode_id"])))
    return out


def load_v4_s3_mixture_plan(receipt_path: Path = DEFAULT_GATE2_RECEIPT) -> dict[str, dict[str, Any]]:
    """Load exact Gate-2-certified `V4-S3` `450M` single-pass water-filling mixture from receipt."""
    data = json.loads(receipt_path.read_text())
    mix = data.get("v4_s3_450m_single_pass_waterfill_mixture", {})
    if not mix:
        raise ValueError(f"Missing `v4_s3_450m_single_pass_waterfill_mixture` in {receipt_path}")
    return {str(k): dict(v) for k, v in mix.items()}


def slice_v4_shard_window(
    shard: dict[str, np.ndarray],
    start: int,
    seq_len: int,
    *,
    excluded_local_ep_ids: set[int] | None = None,
    spec: SLWMv4HeartbeatSpec = SLWMv4HeartbeatSpec(),
) -> dict[str, np.ndarray]:
    """Extract one fixed-length `[seq_len]` dense causal training window from a compact `.npz` shard.

    Dequantizes ONLY the video patches referenced inside `[start : start + seq_len]` (`O(T)` memory!),
    expands per-episode `action_mask`, `proprio_mask`, and `episode_success`, sets `reset_mask[0] = 1`
    and `dt_s[0] = 0.0` cleanly at window start, and pads any short tail with `valid_mask = target_valid_mask = 0`.
    """
    n_tot = int(shard["modality_id"].size)
    end = min(n_tot, start + seq_len)
    n_real = max(0, end - start)

    mod_id = np.zeros((seq_len,), dtype=np.int32)
    hb_id = np.zeros((seq_len,), dtype=np.int32)
    dt_s = np.zeros((seq_len,), dtype=np.float32)
    reset_m = np.zeros((seq_len,), dtype=np.bool_)
    valid_m = np.zeros((seq_len,), dtype=np.float32)
    targ_m = np.zeros((seq_len,), dtype=np.float32)
    rel = np.ones((seq_len,), dtype=np.float32)
    sp_pos = np.zeros((seq_len,), dtype=np.int32)
    txt_b = np.zeros((seq_len,), dtype=np.int32)
    vid_f = np.zeros((seq_len, spec.video_patch_dim), dtype=np.float32)
    aud_c = np.zeros((seq_len, spec.audio_codebooks_stored), dtype=np.int32)
    act_f = np.zeros((seq_len, UNIFIED_ACTION_DIM), dtype=np.float32)
    prop_f = np.zeros((seq_len, UNIFIED_PROPRIO_DIM), dtype=np.float32)
    act_m = np.zeros((seq_len, UNIFIED_ACTION_DIM), dtype=np.float32)
    prop_m = np.zeros((seq_len, UNIFIED_PROPRIO_DIM), dtype=np.float32)
    ep_succ = np.full((seq_len,), -1, dtype=np.int32)

    if n_real <= 0:
        reset_m[0] = True
    else:
        sl = slice(start, end)
        mod_id[:n_real] = shard["modality_id"][sl].astype(np.int32)
        hb_id[:n_real] = shard["heartbeat_id"][sl].astype(np.int32)
        dt_s[:n_real] = np.maximum(shard["dt_s"][sl].astype(np.float32), 0.0)
        reset_m[:n_real] = shard["reset_mask"][sl].astype(np.bool_)
        valid_m[:n_real] = shard["valid_mask"][sl].astype(np.float32)
        targ_m[:n_real] = shard["target_valid_mask"][sl].astype(np.float32)
        rel[:n_real] = shard["reliability"][sl].astype(np.float32)
        sp_pos[:n_real] = shard["spatial_pos"][sl].astype(np.int32)
        txt_b[:n_real] = shard["text_byte"][sl].astype(np.int32)

        # Clean boundary start: never carry stale predecessor state/dt across shard/window cuts!
        reset_m[0] = True
        dt_s[0] = 0.0
        # Clean tail boundary: prevent spurious transition from final real packet into zero-padding!
        if n_real < seq_len:
            reset_m[n_real] = True

        ep_loc = shard["episode_local_id"][sl].astype(np.int32)
        if excluded_local_ep_ids:
            excl_mask = np.isin(ep_loc, np.fromiter(excluded_local_ep_ids, dtype=np.int32))
            if np.any(excl_mask):
                valid_m[:n_real][excl_mask] = 0.0
                targ_m[:n_real][excl_mask] = 0.0

        # Expand per-episode masks & episode_success cleanly
        act_m[:n_real] = shard["episode_action_masks"][ep_loc].astype(np.float32)
        prop_m[:n_real] = shard["episode_proprio_masks"][ep_loc].astype(np.float32)
        ep_succ[:n_real] = shard["episode_success"][ep_loc].astype(np.int32)

        # Dequantize ONLY referenced video patches inside this window!
        v_idx = shard["video_idx"][sl].astype(np.int32)
        v_pos = np.nonzero(v_idx >= 0)[0]
        if v_pos.size > 0:
            v_sel = v_idx[v_pos]
            if "video_q_int8" in shard:
                vid_f[v_pos] = dequantize_wan21_patches_int8(
                    shard["video_q_int8"][v_sel],
                    shard["video_xmin_fp16"][v_sel],
                    shard["video_scale_fp16"][v_sel],
                )
            elif "video_feat" in shard:
                vid_f[v_pos] = shard["video_feat"][v_sel].astype(np.float32)

        a_idx = shard["audio_idx"][sl].astype(np.int32)
        a_pos = np.nonzero(a_idx >= 0)[0]
        if a_pos.size > 0:
            a_sel = a_idx[a_pos]
            aud_c[a_pos] = shard["audio_codes"][a_sel, : spec.audio_codebooks_stored].astype(np.int32)

        m_idx = shard["motor_idx"][sl].astype(np.int32)
        m_pos = np.nonzero(m_idx >= 0)[0]
        if m_pos.size > 0:
            m_sel = m_idx[m_pos]
            act_f[m_pos] = shard["action_feat"][m_sel].astype(np.float32)
            prop_f[m_pos] = shard["proprio_feat"][m_sel].astype(np.float32)

    return {
        "modality_id": mod_id,
        "heartbeat_id": hb_id,
        "dt_s": dt_s,
        "reset_mask": reset_m,
        "valid_mask": valid_m,
        "target_valid_mask": targ_m,
        "reliability": rel,
        "spatial_pos": sp_pos,
        "text_byte": txt_b,
        "video_feat": vid_f,
        "audio_codes": aud_c,
        "action_feat": act_f,
        "proprio_feat": prop_f,
        "action_mask": act_m,
        "proprio_mask": prop_m,
        "episode_success": ep_succ,
    }


def _collate_time_major(windows: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Stack `B` windows `[T, ...]` into time-major `[T, B, ...]` batch arrays."""
    return {k: np.stack([w[k] for w in windows], axis=1) for k in windows[0]}


class _SinglePassCorpusStream:
    """Strictly single-pass (`<= 1.0 epoch`) shard cursor for ONE corpus & ONE split (`train`/`dev`/`test`)."""

    def __init__(
        self,
        *,
        corpus_name: str,
        split: str,
        shard_specs: list[dict[str, Any]],
        seq_len: int,
        packet_cap: int | None,
        excluded_ids: set[tuple[str, str, str]],
        cache_dir: Path | None,
        delete_cached_after_consume: bool,
        seed: int,
        shuffle: bool = True,
    ) -> None:
        self.corpus_name = corpus_name
        self.split = split
        self.seq_len = int(seq_len)
        self.packet_cap = int(packet_cap) if packet_cap is not None else None
        self.excluded_ids = excluded_ids
        self.cache_dir = cache_dir
        self.delete_cached_after_consume = delete_cached_after_consume

        # Validate strict split filename isolation BEFORE touching a single byte!
        expected_prefix = f"{split}_shard_"
        for ss in shard_specs:
            fname = Path(str(ss["path"])).name
            if not fname.startswith(expected_prefix) or not fname.endswith(".npz"):
                raise ValueError(
                    f"Split isolation violation for {corpus_name}/{split}: unexpected shard filename {fname!r}"
                )

        if shuffle and len(shard_specs) > 1:
            rng = np.random.default_rng(seed)
            order = rng.permutation(len(shard_specs)).tolist()
        else:
            order = list(range(len(shard_specs)))
        self._shards = [shard_specs[i] for i in order]
        self._shard_idx = 0
        self._cur_data: dict[str, np.ndarray] | None = None
        self._cur_local_path: Path | None = None
        self._cur_is_temp_download = False
        self._cur_excluded_local: set[int] = set()
        self._cur_pos = 0
        self._cur_len = 0
        self.packets_yielded = 0
        self.exhausted = len(self._shards) == 0

    def _close_current(self) -> None:
        self._cur_data = None
        if self._cur_is_temp_download and self.delete_cached_after_consume and self._cur_local_path is not None:
            try:
                self._cur_local_path.unlink(missing_ok=True)
            except OSError:
                pass
        self._cur_local_path = None
        self._cur_is_temp_download = False
        self._cur_excluded_local = set()
        self._cur_pos = 0
        self._cur_len = 0

    def _advance_shard(self) -> bool:
        self._close_current()
        if self._shard_idx >= len(self._shards):
            self.exhausted = True
            return False

        ss = self._shards[self._shard_idx]
        self._shard_idx += 1

        meta_path: Path | None = None
        if ss["source"] == "local":
            lp = Path(ss["path"])
            self._cur_local_path = lp
            self._cur_is_temp_download = False
            cand_meta = lp.with_name(lp.name.replace(".npz", "_meta.json"))
            if cand_meta.is_file():
                meta_path = cand_meta
        elif ss["source"] == "hf":
            from huggingface_hub import hf_hub_download

            repo_id = str(ss["repo_id"])
            rel_npz = str(ss["path"])
            rel_meta = rel_npz.replace(".npz", "_meta.json")
            dl_kwargs: dict[str, Any] = {"repo_type": "dataset"}
            if self.cache_dir is not None:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                dl_kwargs["local_dir"] = str(self.cache_dir)
            lp = Path(hf_hub_download(repo_id, rel_npz, **dl_kwargs))
            self._cur_local_path = lp
            self._cur_is_temp_download = self.cache_dir is not None
            try:
                meta_path = Path(hf_hub_download(repo_id, rel_meta, **dl_kwargs))
            except Exception as exc:
                if self.split in ("dev", "test"):
                    raise RuntimeError(
                        f"Failed to download mandatory {self.split!r} metadata {repo_id}/{rel_meta} "
                        f"(required for Gate-2 exclusion enforcement): {exc}"
                    ) from exc
                meta_path = None
        else:
            raise ValueError(f"Unknown shard source={ss['source']!r}")

        excl_local: set[int] = set()
        if meta_path is not None and meta_path.is_file():
            try:
                mdata = json.loads(meta_path.read_text())
                for loc_i, rec in enumerate(mdata.get("episodes", [])):
                    key = (self.corpus_name, self.split, str(rec.get("episode_id", "")))
                    if key in self.excluded_ids:
                        excl_local.add(loc_i)
            finally:
                if self._cur_is_temp_download and self.delete_cached_after_consume:
                    meta_path.unlink(missing_ok=True)

        with np.load(self._cur_local_path) as z:
            self._cur_data = {k: z[k] for k in z.files}
        self._cur_excluded_local = excl_local
        self._cur_pos = 0
        self._cur_len = int(self._cur_data["modality_id"].size)
        return True

    def next_window(self) -> dict[str, np.ndarray] | None:
        if self.exhausted:
            return None
        if self.packet_cap is not None and self.packets_yielded >= self.packet_cap:
            self._close_current()
            self.exhausted = True
            return None

        while self._cur_data is None or self._cur_pos >= self._cur_len:
            if not self._advance_shard():
                return None

        win = slice_v4_shard_window(
            self._cur_data,
            self._cur_pos,
            self.seq_len,
            excluded_local_ep_ids=self._cur_excluded_local,
        )
        n_consumed = min(self.seq_len, self._cur_len - self._cur_pos)
        self._cur_pos += n_consumed
        self.packets_yielded += int(np.sum(win["valid_mask"] > 0.5))
        if self.packet_cap is not None and self.packets_yielded >= self.packet_cap:
            # Close cleanly after returning this final window on next call
            pass
        return win

    def get_state(self) -> dict[str, Any]:
        """Serialize exact single-pass shard & packet cursor for zero-revisit preemption recovery."""
        return {
            "shards": list(self._shards),
            "shard_idx": int(self._shard_idx),
            "cur_pos": int(self._cur_pos),
            "has_open_shard": self._cur_data is not None,
            "packets_yielded": int(self.packets_yielded),
            "exhausted": bool(self.exhausted),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore exact single-pass shard & packet cursor without re-reading consumed packets."""
        self._close_current()
        self._shards = list(state["shards"])
        self.packets_yielded = int(state["packets_yielded"])
        self.exhausted = bool(state["exhausted"])
        target_shard_idx = int(state["shard_idx"])
        target_pos = int(state["cur_pos"])
        has_open = bool(state.get("has_open_shard", False))
        if self.exhausted:
            self._shard_idx = len(self._shards)
            return
        if has_open and target_shard_idx > 0:
            self._shard_idx = target_shard_idx - 1
            if self._advance_shard():
                self._cur_pos = min(target_pos, self._cur_len)
        else:
            self._shard_idx = target_shard_idx
            self._cur_pos = 0

    def close(self) -> None:
        self._close_current()


class V4SinglePassMixtureLoader:
    """Deterministic, single-pass (`<= 1.0 epoch`) multimodal batch iterator across SLWM-v4 public corpora.

    Supports both:
      - `local_root`: directory containing `<corpus>/<split>_shard_*.npz` (fast offline & unit test mode).
      - HuggingFace streaming from `valueai-org/slwm-v4-*` public repos (`local_root=None`) with automatic
        bounded local shard cache rotation.
    """

    def __init__(
        self,
        *,
        split: str = "train",
        seq_len: int = 1024,
        batch_size: int = 8,
        seed: int = 42,
        shuffle: bool | None = None,
        local_root: Path | None = None,
        cache_dir: Path | None = Path("/tmp/slwm_v4_shard_cache"),
        delete_cached_after_consume: bool = True,
        mixture_weights: dict[str, float] | None = None,
        packet_caps: dict[str, int] | None = None,
        corpus_repos: dict[str, str] | None = None,
        gate2_receipt_path: Path = DEFAULT_GATE2_RECEIPT,
        exclusion_manifest_path: Path = DEFAULT_EXCLUSION_MANIFEST,
    ) -> None:
        if split not in ("train", "dev", "test"):
            raise ValueError(f"Invalid split={split!r}; expected 'train', 'dev', or 'test'")
        self.split = split
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(seed)
        self.excluded_ids = load_excluded_episode_ids(exclusion_manifest_path)

        repos = dict(V4_TRAIN_CORPUS_REPOS if corpus_repos is None else corpus_repos)
        if mixture_weights is None:
            mix_plan = load_v4_s3_mixture_plan(gate2_receipt_path)
            # Use exact integer `alloc_packets` (zero rounding drift vs 2-decimal `share_pct`)!
            weights = {c: float(mix_plan[c]["alloc_packets"]) for c in repos if c in mix_plan}
            caps = (
                {c: int(mix_plan[c]["alloc_packets"]) for c in repos if c in mix_plan}
                if (packet_caps is None and split == "train")
                else (packet_caps or {})
            )
        else:
            weights = {str(k): float(v) for k, v in mixture_weights.items() if float(v) > 0.0}
            caps = dict(packet_caps or {})

        self.weights = weights
        self.streams: dict[str, _SinglePassCorpusStream] = {}
        self.active_corpora: list[str] = []

        do_shuffle = (split == "train") if shuffle is None else bool(shuffle)
        for ci, (cname, w) in enumerate(weights.items()):
            if w <= 0.0:
                continue
            shards = self._discover_shards(cname, repos.get(cname), local_root)
            if not shards:
                continue
            st = _SinglePassCorpusStream(
                corpus_name=cname,
                split=split,
                shard_specs=shards,
                seq_len=seq_len,
                packet_cap=caps.get(cname),
                excluded_ids=self.excluded_ids,
                cache_dir=cache_dir,
                delete_cached_after_consume=delete_cached_after_consume,
                seed=seed + 1009 * (ci + 1),
                shuffle=do_shuffle,
            )
            self.streams[cname] = st
            self.active_corpora.append(cname)

        if not self.active_corpora:
            raise RuntimeError(f"No valid {split!r} shards discovered (local_root={local_root!r}).")

    def _discover_shards(self, corpus_name: str, repo_id: str | None, local_root: Path | None) -> list[dict[str, Any]]:
        prefix = f"{self.split}_shard_"
        if local_root is not None:
            cdir = Path(local_root) / corpus_name
            if not cdir.is_dir():
                return []
            files = sorted(p for p in cdir.glob(f"{prefix}*.npz") if p.is_file())
            return [{"source": "local", "path": str(p)} for p in files]

        if not repo_id:
            return []
        from huggingface_hub import HfApi

        api = HfApi()
        info = api.repo_info(repo_id, repo_type="dataset")
        rel_files = sorted(
            s.rfilename
            for s in (info.siblings or [])
            if s.rfilename.startswith(f"{corpus_name}/{prefix}") and s.rfilename.endswith(".npz")
        )
        return [{"source": "hf", "repo_id": repo_id, "path": rf} for rf in rel_files]

    def stats(self) -> dict[str, Any]:
        """Return exact single-pass packet exposure telemetry per corpus."""
        by_c = {
            c: {
                "packets_yielded": int(st.packets_yielded),
                "packet_cap": st.packet_cap,
                "exhausted": bool(st.exhausted),
            }
            for c, st in self.streams.items()
        }
        return {
            "split": self.split,
            "total_packets_yielded": sum(v["packets_yielded"] for v in by_c.values()),
            "by_corpus": by_c,
        }

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        return self

    def __next__(self) -> dict[str, np.ndarray]:
        windows: list[dict[str, np.ndarray]] = []
        while len(windows) < self.batch_size:
            live = [c for c in self.active_corpora if not self.streams[c].exhausted]
            if not live:
                if not windows:
                    raise StopIteration
                # Pad remaining batch lanes with invalid zero windows so static `[T, B]` XLA shape is preserved!
                pad_w = {k: np.zeros_like(v) for k, v in windows[0].items()}
                pad_w["reset_mask"][0] = True
                pad_w["episode_success"][:] = -1
                while len(windows) < self.batch_size:
                    windows.append({k: v.copy() for k, v in pad_w.items()})
                break
            probs = np.asarray([self.weights[c] for c in live], dtype=np.float64)
            probs /= float(np.sum(probs))
            chosen = live[int(self.rng.choice(len(live), p=probs))]
            win = self.streams[chosen].next_window()
            if win is not None:
                windows.append(win)
        return _collate_time_major(windows)

    def get_state(self) -> dict[str, Any]:
        """Serialize deterministic RNG + per-corpus single-pass cursors for exact preemption resume."""
        return {
            "split": self.split,
            "rng_state": self.rng.bit_generator.state,
            "streams": {c: st.get_state() for c, st in self.streams.items()},
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore deterministic RNG + per-corpus single-pass cursors cleanly (never revisits consumed packets)."""
        if state.get("split") != self.split:
            raise ValueError(f"Loader split mismatch: checkpoint has {state.get('split')!r}, loader has {self.split!r}")
        self.rng.bit_generator.state = state["rng_state"]
        for c, st_state in state.get("streams", {}).items():
            if c in self.streams:
                self.streams[c].load_state(st_state)

    def close(self) -> None:
        for st in self.streams.values():
            st.close()
