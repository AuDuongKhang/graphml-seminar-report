"""Chạy EVIA trên nhiều bài (lưu từng bài, chạy lại thì bỏ qua bài đã xong) và chấm điểm từ log đã lưu."""
import json
from pathlib import Path

from .evia import ablation_scores, author_scores, run_extraction, run_validation, select
from .metrics import set_prf, universe_metrics
from .prompts import SUBJECT_NAME


def run_evia(lessons: list, clients: dict, models: list, extractors: list, K: int, run_dir: Path,
             workers: int = 4, progress: bool = True) -> Path:
    d = Path(run_dir) / "lessons"
    d.mkdir(parents=True, exist_ok=True)
    it = lessons
    if progress:
        from tqdm.auto import tqdm
        it = tqdm(lessons, desc="EVIA")
    for lesson in it:
        f = d / f"{lesson.name}.json"
        if f.exists():
            continue
        ext = run_extraction(lesson, clients, extractors, workers)
        votes = run_validation(ext["candidates"], clients, models, SUBJECT_NAME[lesson.subject], K, workers)
        f.write_text(json.dumps({**ext, "votes": votes}, ensure_ascii=False), encoding="utf-8")
    return Path(run_dir)


def load_run(run_dir: Path) -> dict:
    return {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in sorted((Path(run_dir) / "lessons").glob("*.json"))}


def run_predictions(run: dict, models: list, K: int, alpha: float, variant: str):
    """-> (dự đoán theo bài, điểm theo bài, pool theo bài, trạng thái các lời gọi trích xuất)."""
    pred, scores, pool, statuses = {}, {}, {}, []
    for name, r in run.items():
        sc = ablation_scores(variant, r["candidates"], r["per_model"], r["votes"], models, K)
        pred[name] = set().union(*select(sc, alpha).values())
        flat = {}
        for seg in sc.values():
            for c, s in seg.items():
                flat[c] = max(flat.get(c, 0.0), s)
        scores[name] = flat
        pool[name] = {c for cs in r["candidates"].values() for c in cs}
        statuses += [s for seg in r.get("status", {}).values() for s in seg.values()]
    return pred, scores, pool, statuses


def score_run(run_dir: Path, lessons: list, models: list, K: int, alpha: float, variant: str = "full") -> dict:
    run = load_run(run_dir)
    gold = {l.name: l.gt for l in lessons if l.name in run}
    pred, scores, pool, _ = run_predictions({k: v for k, v in run.items() if k in gold}, models, K, alpha, variant)
    return {**set_prf(pred, gold), **universe_metrics(pred, gold, pool, scores), "n_lessons": len(gold)}


def score_authors(lessons: list, alpha: float) -> dict:
    """Chấm lại đầu ra LLM của chính tác giả (_Validationner: điểm phiếu sau 5 vòng × 3 LLM)."""
    have = [l for l in lessons if l.author_validation]
    gold = {l.name: l.gt for l in have}
    pred, scores, pool = {}, {}, {}
    for l in have:
        sc = author_scores(l)
        pred[l.name] = set().union(*select(sc, alpha).values())
        scores[l.name] = {c: s for seg in sc.values() for c, s in seg.items()}
        pool[l.name] = set(scores[l.name])
    return {**set_prf(pred, gold), **universe_metrics(pred, gold, pool, scores), "n_lessons": len(gold)}
