"""Chuỗi EVIA (Extraction - Verification - Integration) của SciMKG, ghi lại phiếu theo từng (LLM, vòng).

Khác mã tác giả (findings §B2): điểm tích hợp được chuẩn hóa = số lần sống sót / (|M|·K), nên ngưỡng α có
nghĩa "tần suất" như bài mô tả. Nhờ log phiếu đầy đủ, Table 5 (K × α) và các ablation tính lại offline.
"""
from concurrent.futures import ThreadPoolExecutor

from .llm import parse_concepts
from .normalize import normalize_concept
from .prompts import SUBJECT_NAME, extract_prompt, feedback_prompt, refine_prompt


def _uniq(seq):
    seen, out = set(), []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _concepts(reply):
    items, status = parse_concepts(reply.text)
    return _uniq(normalize_concept(c) for c in items), (reply.error or status)


def run_extraction(lesson, clients: dict, extractors: list, workers: int = 4) -> dict:
    """Mỗi đoạn được mọi extractor trích; ứng viên của đoạn = hợp các kết quả trừ khái niệm đã gặp ở đoạn trước
    (khái niệm gắn với lần xuất hiện đầu tiên, như mã gốc)."""
    subject = SUBJECT_NAME.get(lesson.subject, lesson.subject)
    jobs = [(t, text, m) for t, text in lesson.segments for m in extractors]

    def work(job):
        t, text, m = job
        return job, _concepts(clients[m].chat(extract_prompt(text, subject)))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = dict(ex.map(work, jobs))
    per_model, status, candidates, seen = {}, {}, {}, set()
    for t, text in lesson.segments:
        per_model[t] = {m: results[(t, text, m)][0] for m in extractors}
        status[t] = {m: results[(t, text, m)][1] for m in extractors}
        union = _uniq(c for m in extractors for c in per_model[t][m])
        candidates[t] = [c for c in union if c not in seen]
        seen.update(candidates[t])
    return {"candidates": candidates, "per_model": per_model, "status": status}


def run_validation(candidates: dict, clients: dict, validators: list, subject: str, K: int = 5,
                   workers: int = 4) -> dict:
    """Self-Refine K vòng cho từng (đoạn, LLM): phản hồi (tích lũy) -> lọc. Ghi danh sách sống sót mỗi vòng."""
    jobs = [(t, m) for t, cs in candidates.items() if cs for m in validators]

    def work(job):
        t, m = job
        cs, history, rounds = candidates[t], "hello, ", []
        allowed = set(cs)
        for _ in range(K):
            fb = clients[m].chat(feedback_prompt(cs, subject, history)).text or ""
            kept, _ = _concepts(clients[m].chat(refine_prompt(cs, subject, fb)))
            rounds.append([c for c in kept if c in allowed])
            history += fb + "\n"
        return job, rounds

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = dict(ex.map(work, jobs))
    votes = {}
    for (t, m), rounds in results.items():
        votes.setdefault(t, {})[m] = rounds
    return votes


def integrate(candidates: dict, votes: dict, models: list, K: int) -> dict:
    """Điểm = số (LLM, vòng ≤ K) giữ khái niệm / (len(models)·K)."""
    out = {}
    for t, cs in candidates.items():
        scores = {}
        for c in cs:
            n = sum(c in rnd for m in models for rnd in votes.get(t, {}).get(m, [])[:K])
            scores[c] = n / (len(models) * K)
        out[t] = scores
    return out


def select(scores: dict, alpha: float) -> dict:
    return {t: {c for c, s in sc.items() if s >= alpha} for t, sc in scores.items()}


def ablation_scores(variant: str, candidates: dict, per_model: dict, votes: dict, models: list, K: int) -> dict:
    """full | wo_MLs (chỉ LLM đầu) | wo_SFFD (bỏ tự phản hồi: tỉ lệ LLM trích ra) | wo_both (LLM đầu, trích trực tiếp)."""
    first = models[0]
    if variant == "full":
        return integrate(candidates, votes, models, K)
    if variant == "wo_MLs":
        own = {t: [c for c in candidates[t] if c in set(per_model.get(t, {}).get(first, []))] for t in candidates}
        return integrate(own, votes, [first], K)
    if variant == "wo_SFFD":
        return {t: {c: sum(c in per_model.get(t, {}).get(m, []) for m in models) / len(models) for c in candidates[t]}
                for t in candidates}
    if variant == "wo_both":
        return {t: {c: 1.0 for c in candidates[t] if c in set(per_model.get(t, {}).get(first, []))} for t in candidates}
    raise ValueError(variant)


def author_scores(lesson) -> dict:
    """Điểm phiếu trong _Validationner của tác giả (khởi tạo 1, +1 mỗi lần sống sót, tối đa 1 + 3·5 = 16) -> [0, 1]."""
    return {t: {normalize_concept(c): (s - 1) / 15 for c, s in sc.items()} for t, sc in lesson.author_validation.items()}
