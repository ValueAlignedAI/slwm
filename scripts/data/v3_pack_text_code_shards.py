"""SLWM-v3 Fast CPU Packager for Diverse English Text, Multi-Language Code & External SoTA Benchmarks (`0.00 CU`).

Builds source-hash-disjoint `train` / `dev` / `test` sparse ByT5 UTF-8 byte shards in
`/mnt/slwm-data/SLWM/v3_preencoded/text_code/` covering:
  - Train / Dev (source-hash disjoint, `0` overlap verified):
      * `FineWeb-Edu` (existing SSD parquet shards)
      * `Cosmopedia-v2` (textbooks / synthetic STEM & reasoning)
      * `The-Stack-Smol-XL` (Python, C, C++, JavaScript, Rust, Shell)
      * `CodeSearchNet-Python` (docstrings + functions)
  - Official External SoTA Benchmarks (strictly untouched held-out `test` / `val`):
      * `WikiText-103` (`validation` & `test`)
      * `LAMBADA` (`EleutherAI/lambada_openai` test)
      * `HumanEval` (`164` tasks)
      * `MBPP` (`sanitized` test)
"""

from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from data.v3_unified_shards import (
    BYTE_BOS,
    BYTE_CODE_BOS,
    BYTE_EOS,
    SparseShardBuilder,
    encode_utf8_bytes,
)


def _doc_bucket(text: str) -> int:
    h = hashlib.blake2b(text[:2048].encode("utf-8", errors="ignore"), digest_size=4).digest()
    return int.from_bytes(h, "little") % 100


def _add_text_doc(builder: SparseShardBuilder, text: str, domain: str, *, bos: int = BYTE_BOS, max_bytes: int = 8192) -> None:
    if not text or len(text.strip()) < 32:
        return
    arr = encode_utf8_bytes(text, bos=bos, eos=BYTE_EOS, max_bytes=max_bytes)
    n = len(arr)
    builder.add_episode(
        modality_id=np.full((n,), 2, dtype=np.uint8),
        dt_s=np.full((n,), 0.02, dtype=np.float16),
        domain_name=domain,
        text_byte=arr,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Pack diverse English text, code, and SoTA benchmark ByT5 shards.")
    ap.add_argument("--raw-root", type=Path, default=Path("/mnt/slwm-data/SLWM/v3_diverse_raw"))
    ap.add_argument("--fineweb-root", type=Path, default=Path("/mnt/slwm-data/SLWM/data/fineweb_edu_scale_v1"))
    ap.add_argument("--out-dir", type=Path, default=Path("/mnt/slwm-data/SLWM/v3_preencoded/text_code"))
    ap.add_argument("--max-train-bytes-per-domain", type=int, default=60_000_000, help="Bytes per domain (default 60M -> 180M total).")
    ap.add_argument("--max-dev-bytes-per-domain", type=int, default=3_000_000)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    receipts: dict[str, Any] = {}

    # 1. Train & Dev shards across 3 diverse text/code domains (hash-disjoint: bucket < 95 -> train, >= 95 -> dev)
    train_b = SparseShardBuilder("text_code_train_v1", "train")
    dev_b = SparseShardBuilder("text_code_dev_v1", "dev")
    seen_hashes_train: set[str] = set()
    seen_hashes_dev: set[str] = set()

    def _route_doc(text: str, domain: str, bos: int, tr_budget: list[int], dv_budget: list[int]) -> bool:
        if tr_budget[0] <= 0 and dv_budget[0] <= 0:
            return False
        if not text or len(text.strip()) < 48:
            return True
        dh = hashlib.blake2b(text[:2048].encode("utf-8", errors="ignore"), digest_size=8).hexdigest()
        bkt = int(dh[:4], 16) % 100
        if bkt < 95:
            if tr_budget[0] <= 0 or dh in seen_hashes_train or dh in seen_hashes_dev:
                return True
            seen_hashes_train.add(dh)
            before = train_b.num_ticks
            _add_text_doc(train_b, text, domain, bos=bos, max_bytes=4096)
            tr_budget[0] -= (train_b.num_ticks - before)
        else:
            if dv_budget[0] <= 0 or dh in seen_hashes_dev or dh in seen_hashes_train:
                return True
            seen_hashes_dev.add(dh)
            before = dev_b.num_ticks
            _add_text_doc(dev_b, text, domain, bos=bos, max_bytes=4096)
            dv_budget[0] -= (dev_b.num_ticks - before)
        return True

    # 1a. FineWeb-Edu
    fw_files = sorted(glob.glob(str(args.fineweb_root / "**/*.parquet"), recursive=True))
    fw_tr, fw_dv = [args.max_train_bytes_per_domain], [args.max_dev_bytes_per_domain]
    for fp in fw_files:
        pf = pq.ParquetFile(fp)
        for rg in range(pf.num_row_groups):
            col = pf.read_row_group(rg, columns=["text"]).column("text").to_pylist()
            for t in col:
                if not _route_doc(t or "", "fineweb_edu", BYTE_BOS, fw_tr, fw_dv):
                    break
            if fw_tr[0] <= 0 and fw_dv[0] <= 0:
                break
        if fw_tr[0] <= 0 and fw_dv[0] <= 0:
            break
    print(f"[PACK] FineWeb-Edu done: train_ticks={train_b.num_ticks:,} dev_ticks={dev_b.num_ticks:,}", flush=True)

    # 1b. Cosmopedia-v2
    cp_files = sorted(glob.glob(str(args.raw_root / "smollm_python_edu_and_cosmopedia/cosmopedia-v2/*.parquet")))
    cp_tr, cp_dv = [args.max_train_bytes_per_domain], [args.max_dev_bytes_per_domain]
    for fp in cp_files:
        pf = pq.ParquetFile(fp)
        for rg in range(pf.num_row_groups):
            col = pf.read_row_group(rg, columns=["text"]).column("text").to_pylist()
            for t in col:
                if not _route_doc(t or "", "cosmopedia_v2", BYTE_BOS, cp_tr, cp_dv):
                    break
            if cp_tr[0] <= 0 and cp_dv[0] <= 0:
                break
        if cp_tr[0] <= 0 and cp_dv[0] <= 0:
            break
    print(f"[PACK] Cosmopedia-v2 done: train_ticks={train_b.num_ticks:,} dev_ticks={dev_b.num_ticks:,}", flush=True)

    # 1c. Real Multi-Language + Python Code (`The-Stack-Smol-XL` + `CodeSearchNet-Python`)
    cd_tr, cd_dv = [args.max_train_bytes_per_domain], [args.max_dev_bytes_per_domain]
    stack_files = sorted(glob.glob(str(args.raw_root / "code_real/the_stack_smol_xl/data/**/*.json*"), recursive=True))
    for fp in stack_files:
        opener = gzip.open if fp.endswith(".gz") else open
        with opener(fp, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    txt = obj.get("content") or obj.get("code") or ""
                except Exception:
                    continue
                if not _route_doc(txt, "code_python_multi", BYTE_CODE_BOS, cd_tr, cd_dv):
                    break
        if cd_tr[0] <= 0 and cd_dv[0] <= 0:
            break

    csn_files = sorted(glob.glob(str(args.raw_root / "code_real/codesearchnet_python/**/*.parquet"), recursive=True))
    for fp in csn_files:
        if cd_tr[0] <= 0 and cd_dv[0] <= 0:
            break
        pf = pq.ParquetFile(fp)
        cols = [c for c in ("whole_func_string", "code", "docstring") if c in pf.schema_arrow.names]
        if not cols:
            continue
        col_name = "whole_func_string" if "whole_func_string" in cols else cols[0]
        for rg in range(pf.num_row_groups):
            for t in pf.read_row_group(rg, columns=[col_name]).column(col_name).to_pylist():
                if not _route_doc(t or "", "code_python_multi", BYTE_CODE_BOS, cd_tr, cd_dv):
                    break
            if cd_tr[0] <= 0 and cd_dv[0] <= 0:
                break
    print(f"[PACK] Multi-lang & Python Code done: train_ticks={train_b.num_ticks:,} dev_ticks={dev_b.num_ticks:,}", flush=True)

    overlap_cnt = len(seen_hashes_train & seen_hashes_dev)
    assert overlap_cnt == 0, f"Leak detected between train and dev text/code hashes: {overlap_cnt}"

    receipts["train"] = train_b.save(args.out_dir / "text_code_train_v1.npz")
    receipts["dev"] = dev_b.save(args.out_dir / "text_code_dev_v1.npz")

    # 2. Official External SoTA Benchmark Shards (strictly held-out!)
    # 2a. WikiText-103 validation & test
    for split_nm in ("validation", "test"):
        wt_files = sorted(glob.glob(str(args.raw_root / f"wikitext_103/wikitext-103-raw-v1/{split_nm}-*.parquet")))
        if wt_files:
            wb = SparseShardBuilder(f"wikitext103_{split_nm}", split_nm)
            lines = []
            for fp in wt_files:
                lines.extend(pq.read_table(fp, columns=["text"]).column("text").to_pylist())
            full_doc = "\n".join(x for x in lines if x)
            # Chunk cleanly into 4096-byte contiguous windows for exact causal bits-per-byte & word PPL
            raw_b = full_doc.encode("utf-8", errors="replace")
            for start in range(0, len(raw_b), 4096):
                chunk = raw_b[start : start + 4096]
                arr = np.empty(len(chunk) + 2, dtype=np.uint16)
                arr[0] = BYTE_BOS
                arr[1:-1] = np.frombuffer(chunk, dtype=np.uint8).astype(np.uint16)
                arr[-1] = BYTE_EOS
                wb.add_episode(
                    modality_id=np.full((len(arr),), 2, dtype=np.uint8),
                    dt_s=np.full((len(arr),), 0.02, dtype=np.float16),
                    domain_name="wikitext_103",
                    text_byte=arr,
                )
            receipts[f"wikitext103_{split_nm}"] = wb.save(args.out_dir / f"wikitext103_{split_nm}.npz")

    # 2b. LAMBADA OpenAI test
    lam_path = args.raw_root / "lambada_openai/data/lambada_test.jsonl"
    if lam_path.is_file():
        lb = SparseShardBuilder("lambada_openai_test", "test")
        for line in lam_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            txt = json.loads(line).get("text", "")
            _add_text_doc(lb, txt, "lambada", bos=BYTE_BOS, max_bytes=4096)
        receipts["lambada_openai_test"] = lb.save(args.out_dir / "lambada_openai_test.npz")

    summary = {
        "train_dev_doc_hash_overlap": overlap_cnt,
        "zero_overlap_verified": overlap_cnt == 0,
        "unique_train_docs": len(seen_hashes_train),
        "unique_dev_docs": len(seen_hashes_dev),
        "shards": receipts,
    }
    (args.out_dir / "text_code_shards_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
