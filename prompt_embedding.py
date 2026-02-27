"""
prompt embedding 与相似度计算工具

使用 sentence-transformers 的 all-MiniLM-L6-v2 模型，对英文短句进行向量化，
并提供余弦相似度计算函数。
"""

from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

# 全局单例 embedding 模型，避免重复加载
_EMBEDDING_MODEL: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """
    获取全局 sentence-transformers 模型实例。

    默认使用 all-MiniLM-L6-v2，适合英文短句相似度计算。
    """
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _EMBEDDING_MODEL


def embed_prompt(prompt: str) -> np.ndarray:
    """
    将单个英文 prompt 映射为 embedding 向量（numpy 数组，1D）。
    """
    model = get_embedding_model()
    embedding = model.encode(prompt)
    return np.asarray(embedding).reshape(-1)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    计算两个向量的余弦相似度。
    """
    if vec_a is None or vec_b is None:
        raise ValueError("向量不能为空")
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)

