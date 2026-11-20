"""SLWM-v3 LibriSpeech `train-clean-100` Causal Mimi 32-RVQ + ByT5 Pre-Encoder (100% Speaker-Disjoint from dev/test).

Pre-encodes up to `--max-utterances` real spoken English utterances from official `LibriSpeech train-clean-100`
(`openslr/librispeech_asr`, `train.clean.100`) into unified sparse `.npz` packets (`[Mimi 32-RVQ @ 12.5 Hz ->
BYTE_MOD_SEP -> UTF-8 transcript]` on 50% of clips and `[UTF-8 transcript -> BYTE_MOD_SEP -> Mimi 32-RVQ]` on
50% of clips for true bidirectional `Speech <-> Text` any-to-any learning!).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from data.v3_unified_shards import BYTE_BOS, BYTE_EOS, BYTE_MOD_SEP, BYTE_PAD, SparseShardBuilder, encode_utf8_bytes
from scripts.data.v3_preencode_av_codecs import CausalCodecEngine, _resample_f32


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-encode LibriSpeech train-clean-100 via frozen Kyutai Mimi 32-RVQ.")
    ap.add_argument("--out-path", type=Path, default=Path("/mnt/slwm-data/SLWM/v3_preencoded/audio_text/librispeech_train_clean_100.npz"))
    ap.add_argument("--max-utterances", type=int, default=4000)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("openslr/librispeech_asr", "clean", split="train.100", streaming=True)
    engine = CausalCodecEngine(device=args.device)
    builder = SparseShardBuilder("librispeech_train_clean_100", "train")

    n_done = 0
    speakers: set[str] = set()
    for row in ds:
        if n_done >= args.max_utterances:
            break
        aud = row["audio"]
        wav = np.asarray(aud["array"], dtype=np.float32)
        sr = int(aud["sampling_rate"])
        text = str(row["text"]).strip()
        if not text or len(wav) < sr // 4:
            continue
        wav24 = _resample_f32(wav, sr, 24000)
        # Cap individual utterances at 16s for dense multimodal packing
        if len(wav24) > 16 * 24000:
            wav24 = wav24[: 16 * 24000]
        codes = engine.encode_audio_mimi_32(wav24)  # [T_a, 32] uint16
        if len(codes) < 4:
            continue
        txt_bytes = encode_utf8_bytes(text.lower(), bos=BYTE_MOD_SEP, eos=BYTE_EOS)
        n_a = len(codes)
        n_t = len(txt_bytes)

        # 75% Audio -> Text (ASR / spoken understanding), 25% Text -> Audio (bidirectional TTS/any-to-any!)
        if (n_done % 4) != 3:
            mod = np.concatenate([np.ones((n_a,), dtype=np.uint8), np.full((n_t,), 2, dtype=np.uint8)])
            dt = np.concatenate([np.full((n_a,), 0.08, dtype=np.float16), np.full((n_t,), 0.02, dtype=np.float16)])
            tb = np.concatenate([np.full((n_a,), BYTE_PAD, dtype=np.uint16), txt_bytes])
        else:
            txt_first = encode_utf8_bytes(text.lower(), bos=BYTE_BOS, eos=BYTE_MOD_SEP)
            n_t = len(txt_first)
            mod = np.concatenate([np.full((n_t,), 2, dtype=np.uint8), np.ones((n_a,), dtype=np.uint8)])
            dt = np.concatenate([np.full((n_t,), 0.02, dtype=np.float16), np.full((n_a,), 0.08, dtype=np.float16)])
            tb = np.concatenate([txt_first, np.full((n_a,), BYTE_PAD, dtype=np.uint16)])

        builder.add_episode(
            modality_id=mod,
            dt_s=dt,
            domain_name="librispeech",
            text_byte=tb,
            audio_codes=codes,
        )
        speakers.add(str(row.get("speaker_id", "")))
        n_done += 1
        if n_done % 250 == 0:
            print(f"[LIBRISPEECH_TR100] Encoded {n_done}/{args.max_utterances} utterances ({len(speakers)} speakers)...", flush=True)

    rec: dict[str, Any] = builder.save(args.out_path)
    rec["n_unique_speakers"] = len(speakers)
    (args.out_path.parent / "librispeech_train_clean_100_receipt.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec, indent=2), flush=True)


if __name__ == "__main__":
    main()
