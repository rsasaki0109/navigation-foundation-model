"""Tiny word-level tokenizer for navigation instructions."""

from __future__ import annotations

import numpy as np

from . import world as W

VOCAB = ["<pad>", "<unk>"]
for _t in W.TEMPLATES:
    for _w in _t.split():
        if not _w.startswith("{") and _w not in VOCAB:
            VOCAB.append(_w)
for _n, _c, _rgb, _fp in W.OBJECTS:
    for _w in (_n, _c):
        if _w not in VOCAB:
            VOCAB.append(_w)
WORD2ID = {w: i for i, w in enumerate(VOCAB)}
MAX_TOKENS = 8


def tokenize(text: str) -> np.ndarray:
    ids = [WORD2ID.get(w, 1) for w in text.lower().split()][:MAX_TOKENS]
    return np.array(ids + [0] * (MAX_TOKENS - len(ids)), np.int64)
