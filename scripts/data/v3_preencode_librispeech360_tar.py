"""SLWM-v3 Local LibriSpeech `train-clean-360.tar.gz` Causal Mimi 32-RVQ Pre-Encoder (`0.00 CU`).

Encodes up to `--max-utterances` (default `20,000`, spanning hundreds of unique speakers — 100%
speaker-disjoint from official `dev-clean` & `test-clean`!) directly from `train-clean-360.tar.gz`
using frozen `Kyutai Mimi 32-RVQ` on CUDA / Apple Silicon `mps` / CPU into a compact compressed
sparse `.npz` shard (`75% Audio -> Text`, `25% Text -> Audio` for true bidirectional speech modeling).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tarfile
import time
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.v3_unified_shards import BYTE_BOS, BYTE_EOS, BYTE_MOD_SEP, BYTE_PAD, SparseShardBuilder, encode_utf8_bytes
from scripts.data.v3_preencode_av_codecs import CausalCodecEngine, _decode_audio_bytes_ffmpeg_24k


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-encode LibriSpeech train-clean-360.tar.gz via frozen Kyutai Mimi 32-RVQ.")
    ap.add_argument("--tar-path", type=Path, required=True, help="Path to local `train-clean-360.tar.gz`.")
    ap.add_argument("--out-path", type=Path, required=True, help="Output `.npz` path.")
    ap.add_argument("--max-utterances", type=int, default=20000)
    ap.add_argument("--max-per-speaker", type=int, default=45, help="Cap utterances/speaker to maximize speaker diversity!")
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    t0 = time.perf_counter()
    engine = CausalCodecEngine(device=args.device)
    builder = SparseShardBuilder("librispeech_train_clean_360", "train")

    # Single-pass streaming over tar.gz: buffer each chapter's `.flac` until `.trans.txt` arrives (or vice versa)
    trans_map: dict[str, str] = {}
    pending_flac: dict[str, tuple[str, bytes]] = {}
    spk_counts: dict[str, int] = {}
    n_done = 0

    def _process_pair(utt_id: str, spk_id: str, raw_flac: bytes, text: str) -> None:
        nonlocal n_done
        if n_done >= args.max_utterances:
            return
        if spk_counts.get(spk_id, 0) >= args.max_per_speaker:
            return
        text_c = text.strip().lower()
        if not text_c:
            return
        wav24 = _decode_audio_bytes_ffmpeg_24k(raw_flac)
        if wav24.size < 6000:
            return
        if wav24.size > 16 * 24000:
            wav24 = wav24[: 16 * 24000]
        codes = engine.encode_audio_mimi_32(wav24)
        if len(codes) < 4:
            return
        n_a = len(codes)
        if (n_done % 4) != 3:
            txt_b = encode_utf8_bytes(text_c, bos=BYTE_MOD_SEP, eos=BYTE_EOS)
            n_t = len(txt_b)
            mod = np.concatenate([np.ones((n_a,), dtype=np.uint8), np.full((n_t,), 2, dtype=np.uint8)])
            dt = np.concatenate([np.full((n_a,), 0.08, dtype=np.float16), np.full((n_t,), 0.02, dtype=np.float16)])
            tb = np.concatenate([np.full((n_a,), BYTE_PAD, dtype=np.uint16), txt_b])
        else:
            txt_b = encode_utf8_bytes(text_c, bos=BYTE_BOS, eos=BYTE_MOD_SEP)
            n_t = len(txt_b)
            mod = np.concatenate([np.full((n_t,), 2, dtype=np.uint8), np.ones((n_a,), dtype=np.uint8)])
            dt = np.concatenate([np.full((n_t,), 0.02, dtype=np.float16), np.full((n_a,), 0.08, dtype=np.float16)])
            tb = np.concatenate([txt_b, np.full((n_a,), BYTE_PAD, dtype=np.uint16)])

        builder.add_episode(
            modality_id=mod,
            dt_s=dt,
            domain_name="librispeech",
            text_byte=tb,
            audio_codes=codes,
        )
        spk_counts[spk_id] = spk_counts.get(spk_id, 0) + 1
        n_done += 1
        if n_done % 500 == 0:
            rate = n_done / max(time.perf_counter() - t0, 1e-3)
            print(
                f"[LIBRI360] {n_done}/{args.max_utterances} utts | {len(spk_counts)} speakers | {rate:.1f} utt/s | elapsed={time.perf_counter()-t0:.1f}s",
                flush=True,
            )

    with tarfile.open(args.tar_path, "r:gz") as tf:
        for member in tf:
            if n_done >= args.max_utterances:
                break
            if not member.isfile():
                continue
            nm = member.name
            if nm.endswith(".trans.txt"):
                f_obj = tf.extractfile(member)
                if f_obj is None:
                    continue
                for ln in f_obj.read().decode("utf-8", errors="ignore").splitlines():
                    parts = ln.strip().split(" ", 1)
                    if len(parts) == 2:
                        uid, txt = parts[0].strip(), parts[1].strip()
                        trans_map[uid] = txt
                        if uid in pending_flac:
                            sp_id, flac_b = pending_flac.pop(uid)
                            _process_pair(uid, sp_id, flac_b, txt)
                            if n_done >= args.max_utterances:
                                break
            elif nm.endswith(".flac"):
                stem = Path(nm).stem
                spk_id = stem.split("-")[0]
                if spk_counts.get(spk_id, 0) >= args.max_per_speaker:
                    continue
                f_obj = tf.extractfile(member)
                if f_obj is None:
                    continue
                raw_b = f_obj.read()
                if stem in trans_map:
                    _process_pair(stem, spk_id, raw_b, trans_map.pop(stem))
                else:
                    pending_flac[stem] = (spk_id, raw_b)

    rec: dict[str, Any] = builder.save(args.out_path)
    rec["n_unique_speakers"] = len(spk_counts)
    rec["speaker_ids"] = sorted(spk_counts.keys())
    rec["wall_sec"] = round(time.perf_counter() - t0, 2)
    (args.out_path.parent / "librispeech_train_clean_360_receipt.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "speaker_ids"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
