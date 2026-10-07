"""Truy hồi đồ thị con 2 bước trên ConceptNet cho một câu hỏi (gộp mọi lựa chọn).

Nút = các khái niệm nằm trên ít nhất một đường đi dài <= 2 cạnh nối một khái niệm câu hỏi (qc)
với một khái niệm đáp án (ac); cạnh = các cạnh trên những đường đi đó. Đây là định nghĩa
"E_x lấy từ mọi đường đi k bước nối hai nút của V_x" trong bài GraphVis (k = 2).

ConceptNet được giữ dưới dạng mảng numpy (CSR) thay vì networkx để vừa RAM Colab/Kaggle miễn phí.
"""
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Đúng thứ tự của GraphVis-code/generate_data.py:15-33 và QA-GNN utils/conceptnet.py (id 0..16)
MERGED_RELATIONS = [
    "antonym", "at_location", "capable_of", "causes", "created_by", "is_a", "desires",
    "has_subevent", "part_of", "has_context", "has_property", "made_of", "not_capable_of",
    "not_desires", "receives_action", "related_to", "used_for",
]
RELATED_TO = MERGED_RELATIONS.index("related_to")
HAS_CONTEXT = MERGED_RELATIONS.index("has_context")
# Quy tắc tỉa của QA-GNN construct_graph(prune=True)
BLACKLIST = {"uk", "us", "take", "make", "object", "person", "people"}


def _csv_relation_id(name: str) -> int:
    """Tên quan hệ trong conceptnet.en.csv ('atlocation') -> id."""
    return _CSV_REL[name]


_CSV_REL = {r.replace("_", ""): i for i, r in enumerate(MERGED_RELATIONS)}


@dataclass
class CPNet:
    id2concept: list
    concept2id: dict
    nbr_indptr: np.ndarray      # CSR hàng xóm vô hướng
    nbr_indices: np.ndarray
    edge_keys: np.ndarray       # khóa u*N+v của cạnh xuôi, đã sắp xếp
    edge_rels: np.ndarray       # id quan hệ tương ứng
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def n(self) -> int:
        return len(self.id2concept)

    def has_node(self, u: int) -> bool:
        return 0 <= u < self.n and self.nbr_indptr[u + 1] > self.nbr_indptr[u]

    def neighbors(self, u: int) -> set:
        if u not in self._cache:
            self._cache[u] = set(self.nbr_indices[self.nbr_indptr[u]:self.nbr_indptr[u + 1]].tolist())
        return self._cache[u]

    def relations(self, u: int, v: int) -> list:
        """Các id quan hệ của cạnh xuôi u->v (theo thứ tự trong file nguồn)."""
        k = u * self.n + v
        lo = np.searchsorted(self.edge_keys, k, side="left")
        hi = np.searchsorted(self.edge_keys, k, side="right")
        return self.edge_rels[lo:hi].tolist()

    @classmethod
    def from_edges(cls, id2concept: list, src, dst, rel) -> "CPNet":
        n = len(id2concept)
        src, dst, rel = (np.asarray(x, dtype=np.int64) for x in (src, dst, rel))
        keys = src * n + dst
        order = np.argsort(keys, kind="stable")
        a = np.concatenate([src, dst])
        b = np.concatenate([dst, src])
        pair = np.unique(a * n + b)
        a, b = pair // n, pair % n
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.add.at(indptr, a + 1, 1)
        indptr = np.cumsum(indptr)
        return cls(id2concept=list(id2concept), concept2id={c: i for i, c in enumerate(id2concept)},
                   nbr_indptr=indptr, nbr_indices=b.astype(np.int32),
                   edge_keys=keys[order], edge_rels=rel[order].astype(np.int8))

    @classmethod
    def from_triples(cls, id2concept: list, triples: list) -> "CPNet":
        """Dựng từ danh sách (đầu, quan hệ dạng 'at_location', đuôi) — dùng cho test."""
        c2i = {c: i for i, c in enumerate(id2concept)}
        r2i = {r: i for i, r in enumerate(MERGED_RELATIONS)}
        s, d, r = zip(*[(c2i[h], c2i[t], r2i[rel]) for h, rel, t in triples])
        return cls.from_edges(id2concept, s, d, r)


def load_cpnet_csv(csv_path: Path, concept_txt: Path, cache: Path | None = None) -> CPNet:
    """Dựng đồ thị đã tỉa giống QA-GNN từ conceptnet.en.csv (rel, đầu, đuôi, trọng số; tách bằng tab)."""
    with open(concept_txt, encoding="utf-8") as f:
        id2concept = [w.strip() for w in f]
    if cache is not None and Path(cache).exists():
        z = np.load(cache)
        return CPNet.from_edges(id2concept, z["src"], z["dst"], z["rel"])
    c2i = {c: i for i, c in enumerate(id2concept)}
    seen, src, dst, rel = set(), [], [], []
    n = len(id2concept)
    with open(csv_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            r = _csv_relation_id(parts[0])
            h, t = parts[1], parts[2]
            if h in BLACKLIST or t in BLACKLIST or r == HAS_CONTEXT:
                continue
            s, d = c2i[h], c2i[t]
            if s == d:
                continue
            key = (s * n + d) * len(MERGED_RELATIONS) + r
            if key in seen:
                continue
            seen.add(key)
            src.append(s)
            dst.append(d)
            rel.append(r)
    if cache is not None:
        np.savez_compressed(cache, src=np.array(src, dtype=np.int32), dst=np.array(dst, dtype=np.int32),
                            rel=np.array(rel, dtype=np.int8))
    return CPNet.from_edges(id2concept, src, dst, rel)


@dataclass
class SubGraph:
    nodes: list
    triples: list
    qc: set
    ac: set


def _pretty(name: str) -> str:
    return name.replace("_", " ")


def _forward_relation(cp: CPNet, u: int, v: int):
    """Quy tắc get_edge của GraphVis: quan hệ không trùng, bỏ related_to nếu còn quan hệ khác, lấy cái đầu."""
    rels = []
    for r in cp.relations(u, v):
        if r not in rels:
            rels.append(r)
    if len(rels) > 1 and RELATED_TO in rels:
        rels.remove(RELATED_TO)
    return rels[0] if rels else None


def _triple(cp: CPNet, a: int, b: int):
    r = _forward_relation(cp, a, b)
    if r is not None:
        return a, r, b
    r = _forward_relation(cp, b, a)
    if r is not None:
        return b, r, a
    return None


def build_question_graph(cp: CPNet, grounded_rows: list, max_nodes: int = 80,
                         max_hub_degree: int = 500, paths_per_pair: int = 1) -> SubGraph:
    """max_hub_degree: bỏ khái niệm có bậc lớn hơn (nút "hub" như human, number);
    paths_per_pair: số nút trung gian giữ lại cho mỗi cặp (q, a), ưu tiên nút bậc thấp (cụ thể nhất).
    Mặc định (500, 1) cho phân bố số nút/cạnh trên CSQA dev gần Fig 4-5 của bài (trung vị ~21 nút, ~26 cạnh)."""
    def deg(u):
        return int(cp.nbr_indptr[u + 1] - cp.nbr_indptr[u])

    def ids(names):
        out = []
        for name in names:
            i = cp.concept2id.get(name)
            if i is not None and cp.has_node(i) and deg(i) <= max_hub_degree:
                out.append(i)
        return out

    qc_all, ac_all = set(), set()
    score = Counter()          # số đường đi đi qua mỗi nút
    edges = set()              # cặp nút vô hướng (min, max) nằm trên một đường đi
    for row in grounded_rows:
        qs, as_ = ids(row.get("qc", [])), ids(row.get("ac", []))
        qc_all.update(qs)
        ac_all.update(as_)
        for q in qs:
            nq = cp.neighbors(q)
            for a in as_:
                if q == a:
                    continue
                if a in nq:
                    score.update([q, a])
                    edges.add((min(q, a), max(q, a)))
                mids = [x for x in nq & cp.neighbors(a) if x not in (q, a) and deg(x) <= max_hub_degree]
                mids.sort(key=lambda x: (deg(x), cp.id2concept[x]))
                for x in mids[:paths_per_pair]:
                    score.update([q, x, a])
                    edges.add((min(q, x), max(q, x)))
                    edges.add((min(x, a), max(x, a)))
    by_score = lambda n: (-score[n], cp.id2concept[n])
    endpoints = sorted((n for n in score if n in qc_all or n in ac_all), key=by_score)
    extras = sorted((n for n in score if n not in qc_all and n not in ac_all), key=by_score)
    kept = (endpoints + extras)[:max_nodes]
    keep = set(kept)
    triples = []
    for a, b in sorted(edges):
        if a in keep and b in keep:
            t = _triple(cp, a, b)
            if t is not None:
                h, r, tl = t
                triples.append((_pretty(cp.id2concept[h]), _pretty(MERGED_RELATIONS[r]), _pretty(cp.id2concept[tl])))
    return SubGraph(nodes=[_pretty(cp.id2concept[n]) for n in kept], triples=triples,
                    qc={_pretty(cp.id2concept[n]) for n in qc_all},
                    ac={_pretty(cp.id2concept[n]) for n in ac_all})
