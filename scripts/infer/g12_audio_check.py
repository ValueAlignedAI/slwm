"""Verify real microphone substitutions through the shipped world model and compare with cached codec outputs."""

import json
from pathlib import Path
import numpy as np
from scripts.infer.g12_live import raw_speech, run


def main():
    root = Path("/mnt/slwm-data/SLWM/g12/release")
    examples = root / "examples"
    package = root / "inference"
    index = json.loads((package / "manifest.json").read_bytes())
    model = Path("/mnt/slwm-data/SLWM/g11/data/whisper-small")
    results = []
    for rec in json.loads((examples / "streams.json").read_bytes()):
        fam = rec["family"]
        event = rec["audio_example"]["index"]
        file = examples / (fam + "-stream.npz")
        with np.load(file) as z:
            data = {k: z[k] for k in z.files}
        for mode, k1, k2 in (("true", "ac", "words"), ("foil", "wc", "wrong")):
            wave = examples / f"{fam}-{mode}.wav"
            ac, words, sentence = raw_speech(wave, model, index["lexicon"])
            err = float(np.mean(np.abs(ac.astype("float32") - data[k1][event])))
            match = float(np.mean(words == data[k2][event]))
            if (
                not np.isfinite(ac).all()
                or not np.isfinite(err)
                or ac.shape != (4, 128)
            ):
                raise ValueError("Invalid microphone codec")
            # Run the actual public interface, including its waveform override and commitment gate.
            actual = next(
                run(
                    file,
                    package,
                    rec["fold"],
                    "associative",
                    "correct",
                    event,
                    wave,
                    model,
                )
            )
            before = next(
                run(
                    file,
                    package,
                    rec["fold"],
                    "associative",
                    "correct" if mode == "true" else "wrong",
                    event,
                )
            )
            results.append(
                dict(
                    family=fam,
                    mode=mode,
                    transcribed=sentence,
                    acoustic_mae=err,
                    text_feature_agreement=match,
                    same_routing=actual["route"] == before["route"],
                    public_output=actual,
                    prepared_output=before,
                )
            )
    (root / "metrics/raw-audio.json").write_text(json.dumps(results, indent=2) + "\n")
    print("Real-waveform inference verified", len(results), flush=True)


if __name__ == "__main__":
    main()
