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


TRANSIENT = {"no_response", "timeout", "api_error"}


def extract_one_model(lesson, client, workers: int = 4):
    """Một mô hình trích xuất mọi đoạn của bài -> ({mốc: [khái niệm]}, {mốc: trạng thái})."""
    subject = SUBJECT_NAME.get(lesson.subject, lesson.subject)

    def work(seg):
        t, text = seg
        return t, _concepts(client.chat(extract_prompt(text, subject)))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = dict(ex.map(work, lesson.segments))
    return {t: r[0] for t, r in res.items()}, {t: r[1] for t, r in res.items()}


def combine_candidates(lesson, per_model: dict, extractors: list) -> dict:
    """Ứng viên của đoạn = hợp kết quả các extractor trừ khái niệm đã gặp ở đoạn trước (gắn lần xuất hiện đầu)."""
    candidates, seen = {}, set()
    for t, _ in lesson.segments:
        union = _uniq(c for m in extractors for c in per_model.get(t, {}).get(m, []))
        candidates[t] = [c for c in union if c not in seen]
        seen.update(candidates[t])
    return candidates


def run_extraction(lesson, clients: dict, extractors: list, workers: int = 4) -> dict:
    per_model, status = {}, {}
    for m in extractors:
        pm, st = extract_one_model(lesson, clients[m], workers)
        for t in pm:
            per_model.setdefault(t, {})[m] = pm[t]
            status.setdefault(t, {})[m] = st[t]
    return {"candidates": combine_candidates(lesson, per_model, extractors), "per_model": per_model, "status": status}


def run_validation(candidates: dict, clients: dict, validators: list, subject: str, K: int = 5,
                   workers: int = 4, statuses: list | None = None) -> dict:
    """Self-Refine K vòng cho từng (đoạn, LLM): phản hồi (tích lũy) -> lọc. Ghi danh sách sống sót mỗi vòng;
    vòng có lời gọi lỗi/sai định dạng ghi None (không phải phiếu chống) và trạng thái được đưa vào `statuses`."""
    statuses = statuses if statuses is not None else []
    jobs = [(t, m) for t, cs in candidates.items() if cs for m in validators]

    def work(job):
        t, m = job
        cs, history, rounds = candidates[t], "hello, ", []
        allowed = set(cs)
        for _ in range(K):
            fb = clients[m].chat(feedback_prompt(cs, subject, history))
            statuses.append(fb.error or "ok")
            if fb.error:
                rounds.append(None)
                continue
            kept, status = _concepts(clients[m].chat(refine_prompt(cs, subject, fb.text)))
            statuses.append(status)
            rounds.append([c for c in kept if c in allowed] if status == "ok" else None)
            history += fb.text + "\n"
        return job, rounds

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = dict(ex.map(work, jobs))
    votes = {}
    for (t, m), rounds in results.items():
        votes.setdefault(t, {})[m] = rounds
    return votes


def integrate(candidates: dict, votes: dict, models: list, K: int) -> dict:
    """Điểm = số (LLM, vòng ≤ K) giữ khái niệm / số (LLM, vòng ≤ K) chạy thành công (vòng None bị loại)."""
    out = {}
    for t, cs in candidates.items():
        rounds = [r for m in models for r in votes.get(t, {}).get(m, [])[:K] if r is not None]
        out[t] = {c: (sum(c in r for r in rounds) / len(rounds) if rounds else 0.0) for c in cs}
    return out


def select(scores: dict, alpha: float) -> dict:
    return {t: {c for c, s in sc.items() if s >= alpha} for t, sc in scores.items()}


def ablation_scores(variant: str, candidates: dict, per_model: dict, votes: dict, models: list, K: int) -> dict:
    """full | wo_MLs (chỉ LLM đầu) | wo_SFFD (bỏ tự phản hồi: tỉ lệ LLM trích ra) | wo_both (LLM đầu, trích trực tiếp)."""
    # mô hình trích xuất thực tế lấy từ log (có thể khác --models khi chạy evia --extractors)
    extractors = sorted({m for seg in per_model.values() for m in seg}) or list(models)
    first = models[0]
    first_ext = first if first in extractors else extractors[0]
    own = {t: [c for c in candidates[t] if c in set(per_model.get(t, {}).get(first_ext, []))] for t in candidates}
    if variant == "full":
        return integrate(candidates, votes, models, K)
    if variant == "wo_MLs":
        return integrate(own, votes, [first], K)
    if variant == "wo_SFFD":
        return {t: {c: sum(c in per_model.get(t, {}).get(m, []) for m in extractors) / len(extractors)
                    for c in candidates[t]} for t in candidates}
    if variant == "wo_both":
        return {t: {c: 1.0 for c in own[t]} for t in candidates}
    raise ValueError(variant)


def author_scores(lesson) -> dict:
    """Điểm phiếu trong _Validationner của tác giả (khởi tạo 1, +1 mỗi lần sống sót, tối đa 1 + 3·5 = 16) -> [0, 1]."""
    return {t: {normalize_concept(c): (s - 1) / 15 for c, s in sc.items()} for t, sc in lesson.author_validation.items()}
