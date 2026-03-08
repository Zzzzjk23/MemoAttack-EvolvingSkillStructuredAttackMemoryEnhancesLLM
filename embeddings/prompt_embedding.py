"""
Prompt embedding and cosine-similarity helpers.

Prefer sentence-transformers when available. Fall back to a deterministic
hash-based embedding so local tests can run without the heavy dependency.
"""

from __future__ import annotations

from hashlib import blake2b
from typing import Optional

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None


_EMBEDDING_MODEL: Optional["SentenceTransformer"] = None
_FALLBACK_DIMENSION = 32


def get_embedding_model() -> Optional["SentenceTransformer"]:
    global _EMBEDDING_MODEL
    if SentenceTransformer is None:
        return None
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _EMBEDDING_MODEL


def _fallback_embed(prompt: str) -> np.ndarray:
    vector = np.zeros(_FALLBACK_DIMENSION, dtype=float)
    for index, token in enumerate(prompt.split()):
        digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest, "big") % _FALLBACK_DIMENSION
        sign = 1.0 if digest[0] % 2 == 0 else -1.0
        vector[bucket] += sign * (1.0 + (index % 3) * 0.1)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def embed_prompt(prompt: str) -> np.ndarray:
    model = get_embedding_model()
    if model is None:
        return _fallback_embed(prompt)
    embedding = model.encode(prompt)
    return np.asarray(embedding).reshape(-1)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    if vec_a is None or vec_b is None:
        raise ValueError("Vectors cannot be None")
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)
