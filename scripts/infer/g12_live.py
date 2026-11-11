"""Replay actual timestamped multimodal signals through the released proposal and commit policy."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
from data.g12_language import encode_words
from models.g12_world import Spec, MODES, proposals, route

PUBLIC = (
    Path(__file__).resolve().parents[2]
    / "experiments/multimodal/t12/instruction-sequences-20261111/inference"
)
KEYS = ("scene", "ac", "wc", "words", "wrong", "human", "wrong_human")


def raw_speech(path, model_path, lexicon):
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    from scripts.data.g11_speech import pool_time

    pcm = np.frombuffer(
        subprocess.check_output(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "f32le",
                "pipe:1",
            ],
            timeout=65,
        ),
        "<f4",
    )
    if not 0.1 <= len(pcm) / 16000 <= 28.1:
        raise ValueError("Expected one completed 0.1–28 s utterance")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    proc = WhisperProcessor.from_pretrained(str(model_path), local_files_only=True)
    net = (
        WhisperForConditionalGeneration.from_pretrained(
            str(model_path), local_files_only=True, dtype=dtype
        )
        .to(dev)
        .eval()
    )
    proj = np.linalg.qr(
        np.random.default_rng(4173).standard_normal((net.config.d_model, 128))
    )[0].astype(np.float32)
    with torch.inference_mode():
        inp = proc.feature_extractor(
            [pcm.copy()],
            sampling_rate=16000,
            return_attention_mask=True,
            return_tensors="pt",
            device=dev,
        )
        x = inp.input_features.to(dev, dtype=dtype)
        h = net.model.encoder(x, return_dict=True)
        seq = net.generate(
            input_features=x,
            encoder_outputs=h,
            attention_mask=inp.attention_mask.to(dev),
            max_new_tokens=96,
            num_beams=1,
            do_sample=False,
        )
        sentence = proc.batch_decode(seq, skip_special_tokens=True)[0].strip()
        emb = (
            h.last_hidden_state[0, : max(1, min(1500, int(np.ceil(len(pcm) / 320))))]
            .float()
            .cpu()
            .numpy()
        )
    acoustic = (
        np.clip(pool_time(emb) @ proj, -20, 20)
        .astype(np.float16)
        .reshape(4, 4, 128)
        .mean(1)
    )
    return acoustic, encode_words(sentence, lexicon), sentence


def load(package, fold, core, width=128, recipe="primary"):
    index = json.loads((package / "manifest.json").read_bytes())
    entry = index["models"][
        fold
        + "-"
        + core
        + ("-early" if recipe == "early" else "-w256" if width == 256 else "")
    ]
    file = package / entry["file"]
    with file.open("rb") as f:
        checksum = hashlib.file_digest(f, "sha256").hexdigest()
    if checksum != entry["sha256"]:
        raise ValueError("Published inference weights failed verification")
    with np.load(file, allow_pickle=False) as z:
        p = {k: jnp.asarray(z[k]) for k in z.files}
    return index, entry, p


def run(
    stream,
    package,
    fold,
    core,
    mode="correct",
    event=None,
    wave=None,
    recognizer=None,
    width=128,
    batch_size=1,
    recipe="primary",
):
    if recipe == "early" and width != 128:
        raise ValueError("Early models have width 128")
    index, entry, p = load(package, fold, core, width, recipe)
    s = Spec(**entry["spec"])
    with np.load(stream, allow_pickle=False) as z:
        features = {k: z[k] for k in KEYS}
        times = z["seconds"]
    if np.any(np.diff(times) <= 0):
        raise ValueError("Live replay requires strictly increasing source clocks")
    selected = np.arange(len(times)) if event is None else np.asarray([event])
    if selected.min() < 0 or selected.max() >= len(times):
        raise ValueError("Event is outside this recording")
    if wave is not None:
        if event is None or recognizer is None or mode != "correct":
            raise ValueError(
                "Use one selected event and a local Whisper-small.en codec for an audio substitution"
            )
        ac, w, _ = raw_speech(wave, recognizer, index["lexicon"])
        features["ac"][event] = ac
        features["words"][event] = w

    @jax.jit
    def forward(params, inputs):
        o = proposals(params, s, core, inputs, mode)
        return (
            jax.nn.softmax(o["verb"], -1),
            jax.nn.sigmoid(o["state"]),
            jax.nn.sigmoid(o["certainty"]),
        )

    if not 1 <= batch_size <= 256:
        raise ValueError("Batch size outside 1–256")
    for offset in range(0, len(selected), batch_size):
        take = selected[offset : offset + batch_size]
        padded = np.pad(take, (0, batch_size - len(take)), mode="edge")
        verb, state, certainty = map(
            np.asarray,
            forward(p, {k: jnp.asarray(v[padded]) for k, v in features.items()}),
        )
        for j, idx in enumerate(take):
            conf = [
                0.5 * (verb[j].max(-1).mean() + certainty[j, 0]),
                0.5 * (max(state[j], 1 - state[j]) + certainty[j, 1]),
            ]
            decision = route(
                verb[j], state[j], conf, entry["thresholds"], entry["vocabulary"]
            )
            yield dict(second=float(times[idx]), **decision)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("stream", type=Path)
    a.add_argument("--package", type=Path, default=PUBLIC)
    a.add_argument("--fold", choices=("A", "B"), required=True)
    a.add_argument(
        "--core",
        choices=("gpt2", "gated", "oscillator", "associative"),
        default="associative",
    )
    a.add_argument("--mode", choices=MODES, default="correct")
    a.add_argument("--event", type=int)
    a.add_argument("--wav", type=Path)
    a.add_argument("--whisper", type=Path)
    a.add_argument("--width", type=int, choices=(128, 256), default=128)
    a.add_argument("--batch-size", type=int, default=1)
    a.add_argument("--recipe", choices=("primary", "early"), default="primary")
    c = a.parse_args()
    for r in run(
        c.stream,
        c.package,
        c.fold,
        c.core,
        c.mode,
        c.event,
        c.wav,
        c.whisper,
        c.width,
        c.batch_size,
        c.recipe,
    ):
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
