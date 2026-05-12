# SLWM Data Contract — Sprint I0

**Sprint:** I0 — repository skeleton and contracts  
**Status:** schema and validation helpers only; no dataset loading or preprocessing.

## 1. Unified Sample Schema

Each raw/preprocessed sample should use the following high-level schema before modality adapters:

```json
{
  "sample_id": "stable-id",
  "streams": {
    "text_code": {"data": "...", "start": 0.0, "end": 5.0},
    "audio": {"path": "...", "start": 0.0, "end": 5.0},
    "visual_video": {"path": "...", "fps": 8, "start": 0.0, "end": 5.0}
  },
  "targets": {
    "future_text": null,
    "future_audio": null,
    "future_video": null,
    "caption": null,
    "answerability": null
  },
  "metadata": {
    "dataset": "...",
    "license": "...",
    "language": "en",
    "split": "train"
  }
}
```

## 2. Required Modality IDs

| Modality | ID | Notes |
|---|---:|---|
| `noop` | 0 | Valid committed policy behavior. |
| `text_code` | 1 | English text and code edge codec. |
| `audio` | 2 | Speech/general audio latent or feature stream. |
| `visual_video` | 3 | Image/video patch, tube, or latent stream. |

Optional action/sensor modalities are future-phase and are not enabled by I0.

## 3. Adapter Output Contract

Every modality adapter maps an input stream into:

```python
{
  "z": FloatTensor[B,T,D],
  "mask": BoolTensor[B,T],
  "metadata": {
    "modality": str,
    "modality_id": int,
    "observed": bool
  }
}
```

The mask distinguishes valid/observed positions from padded or missing positions. Later sprints must preserve observed vs predicted/inferred/imagined metadata to avoid unsupported-output confusion.

## 4. Source Tags

Future uncertainty/source heads and exploration probes must use this controlled tag set:

```text
observed
reconstructed
predicted
inferred
imagined
unknown
unsupported
```

I0 tests only verify that the contract is importable and stable.
