"""KAPING (Baek et al., 2023): chọn k bộ ba gần câu hỏi nhất theo embedding câu rồi chèn dạng văn bản vào prompt.

Bài GraphVis tự cài lại KAPING trên cùng LVLM; mã của họ không công bố. Nhóm dùng encoder
MPNet (`sentence-transformers/all-mpnet-base-v2`, như bài KAPING) và k = 10.
"""
import numpy as np

KAPING_HEADER = "Below are facts in the form of the triple meaningful to answer the question."
DEFAULT_ENCODER = "sentence-transformers/all-mpnet-base-v2"
_MODEL = None


def _default_encode(texts):
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(DEFAULT_ENCODER)
    return _MODEL.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


def top_k_triples(question: str, triples: list, k: int = 10, encode=None) -> list:
    if not triples:
        return []
    encode = encode or _default_encode
    vecs = encode([question] + [f"{h} {r} {t}" for h, r, t in triples])
    scores = vecs[1:] @ vecs[0]
    order = np.argsort(-scores, kind="stable")[:k]
    return [tuple(triples[i]) for i in order]


def kaping_prefix(triples: list) -> str:
    if not triples:
        return ""
    return KAPING_HEADER + "\n" + "".join(f"({h}, {r}, {t})\n" for h, r, t in triples)
