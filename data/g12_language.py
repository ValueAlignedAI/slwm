"""Causal four-slot language codec for completed recognized words."""

import re
import numpy as np
from data.g10_language import text_features


def encode_words(sentence, lexicon):
    tokens = re.findall(r"[a-z0-9']+", sentence.casefold())
    slots = np.zeros((4, 128 + len(lexicon)), np.float16)
    if not tokens:
        return slots
    # Word order remains visible. The last semantic channels are fixed public action-word cues at the edge.
    chunks = np.array_split(np.arange(len(tokens)), min(4, len(tokens)))
    for i, ix in enumerate(chunks):
        s = " ".join(tokens[j] for j in ix)
        slots[i, :128] = text_features(s).astype(np.float32) / 127
        slots[i, 128:] = [
            float(any(re.search(r"\b" + re.escape(x) + r"\b", s) for x in aliases))
            for aliases in (aliases for _, aliases in lexicon)
        ]
    return slots
