"""Encode whole original utterances with pinned Whisper-small.en; retain causal audio from the complete cue."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from data.g14_grid import T
from g14_runtime import ROOT, json_file
from data.g14_wave import pcm


def main():
    inp = ROOT / "data/utterances"
    out = ROOT / "data/semantic-utterances"
    out.mkdir(exist_ok=True)
    media = json.loads((inp / "media.json").read_bytes())
    snapshot = Path("/mnt/slwm-data/SLWM/g11/data/whisper-small")
    proc = WhisperProcessor.from_pretrained(str(snapshot), local_files_only=True)
    enc = (
        WhisperForConditionalGeneration.from_pretrained(
            str(snapshot), local_files_only=True, dtype=torch.float16
        )
        .model.encoder.cuda()
        .eval()
    )
    basis = np.linalg.qr(
        np.random.default_rng(141077).standard_normal((enc.config.d_model, 64))
    )[0].astype("f4")
    project = torch.tensor(basis, device="cuda", dtype=torch.float32)
    step = np.rint((np.arange(T) * 0.032 + 0.016) / 0.020).astype(int)
    with torch.inference_mode(), ThreadPoolExecutor(6) as workers:
        for role in ("train", "dev", "cal", "held", "fresh"):
            source = inp / (role + ".npz")
            dest = out / (role + ".npz")
            if dest.is_file():
                continue
            with np.load(source) as z:
                block = {k: z[k] for k in z.files if k != "a"}
            ids = block["global_index"]
            pieces = []
            tic = time.monotonic()
            for i in range(0, len(ids), 32):
                rows = [media[ix] for ix in ids[i : i + 32]]
                waves = list(workers.map(pcm, rows))
                x = proc.feature_extractor(
                    waves, sampling_rate=16000, return_tensors="pt", device="cuda"
                ).input_features.to("cuda", dtype=torch.float16)
                h = (
                    enc(x, return_dict=True).last_hidden_state[:, step].float()
                    @ project
                )
                pieces.append(h.cpu().numpy().astype("f2"))
                if (i // 32) % 50 == 49:
                    print(role, i + len(rows), len(ids), "complete", flush=True)
            features = np.concatenate(pieces)
            mask = np.arange(T)[None, :] < block["length"][:, None]
            if role == "train":
                valid = features[mask].astype("f4")
                mu = valid.mean(0)
                sd = np.maximum(valid.std(0), 0.08)
                np.savez_compressed(
                    out / "normalizer.npz", mean=mu, std=sd, basis=basis
                )
            else:
                with np.load(out / "normalizer.npz") as z:
                    mu = z["mean"]
                    sd = z["std"]
            block["a"] = (
                np.clip(0.5 + (features.astype("f4") - mu) / (6 * sd), 0, 1)
                * mask[:, :, None]
            ).astype("f2")
            temp = dest.with_suffix(".new.npz")
            np.savez_compressed(temp, **block)
            temp.replace(dest)
            print(
                "Frozen speech states",
                role,
                len(ids),
                round(time.monotonic() - tic, 1),
                flush=True,
            )
    json_file(
        out / "complete.json",
        dict(
            edge="openai/whisper-small.en",
            revision="e8727524f962ee844a7319d92be39ac1bd25655a",
            encoder_parameters=sum(v.numel() for v in enc.parameters()),
            projection_seed=141077,
            clock_ms=32,
        ),
    )


if __name__ == "__main__":
    main()
