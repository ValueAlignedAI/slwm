"""Deterministic causal adapters for completed natural utterances."""
import hashlib
import re
import numpy as np


def text_features(sentence):
    tokens = re.findall(r"[\w']+", sentence.casefold())
    grams = ["w:" + s for s in tokens] + ["b:" + a + " " + b for a,b in zip(tokens,tokens[1:])]
    chars = " "+" ".join(tokens)+" "
    grams += ["c:"+chars[i:i+3] for i in range(len(chars)-2)]
    out = np.zeros(128, np.float32)
    for gram in grams:
        h = int.from_bytes(hashlib.blake2s(gram.encode(),digest_size=4).digest(),"little")
        out[h%128] += 1 if h & 128 else -1
    return np.rint(127*out/max(1.,np.linalg.norm(out))).astype(np.int8)



def timeline(events,ticks,hz=10):
    out=np.zeros((ticks,128),np.int8)
    for ev in sorted(events,key=lambda e:e["end"]):
        first=max(0,int(np.ceil(float(ev["end"])*hz))-1)
        if first<ticks:out[first:min(ticks,first+8*hz)]=text_features(ev["text"])
    return out
