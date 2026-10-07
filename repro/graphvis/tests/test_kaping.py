import numpy as np

from gv.data.kaping import KAPING_HEADER, kaping_prefix, top_k_triples

VOCAB = ["bank", "door", "money", "dog"]


def fake_encode(texts):
    """Vector đếm từ khóa -> cosine đo số từ chung."""
    m = np.array([[t.count(w) for w in VOCAB] for t in texts], dtype=float) + 1e-6
    return m / np.linalg.norm(m, axis=1, keepdims=True)


TRIPLES = [("dog", "capable of", "bark"), ("door", "at location", "bank"), ("money", "at location", "bank")]


def test_top_k_ranks_by_similarity():
    out = top_k_triples("Where is a revolving door and money? bank", TRIPLES, k=2, encode=fake_encode)
    assert ("dog", "capable of", "bark") not in out and len(out) == 2


def test_top_k_returns_all_when_k_large():
    assert len(top_k_triples("bank", TRIPLES, k=10, encode=fake_encode)) == 3


def test_top_k_empty():
    assert top_k_triples("bank", [], k=10, encode=fake_encode) == []


def test_kaping_prefix_format():
    p = kaping_prefix([("door", "at location", "bank")])
    assert p == KAPING_HEADER + "\n(door, at location, bank)\n"
    assert kaping_prefix([]) == ""
