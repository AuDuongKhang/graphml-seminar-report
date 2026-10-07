"""Chạy EVIA trên nhiều bài (lưu từng bài, chạy lại thì bỏ qua bài đã xong) và chấm điểm từ log đã lưu."""
import json
from pathlib import Path

from .evia import (TRANSIENT, ablation_scores, author_scores, combine_candidates, extract_one_model,
                   run_validation, select)
from .metrics import set_prf, universe_metrics
from .prompts import SUBJECT_NAME


def _write_atomic(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def run_evia(lessons: list, clients: dict, models: list, extractors: list, K: int, run_dir: Path,
             workers: int = 4, progress: bool = True, allow_partial: bool = False) -> Path:
    """Chạy lần lượt TỪNG mô hình trên mọi bài (Ollama không phải nạp lại mô hình liên tục).
    Bài có lời gọi lỗi tạm thời (mạng, quá giờ, rỗng) chỉ được lưu dạng *.partial.json; chạy lại sẽ thử lại
    (lời gọi thành công lấy từ cache). allow_partial=True: chấp nhận luôn để không kẹt khi lỗi lặp lại."""
    d = Path(run_dir) / "lessons"
    d.mkdir(parents=True, exist_ok=True)
    todo = [l for l in lessons if not (d / f"{l.name}.json").exists()]
    bar = (lambda xs, desc: xs)
    if progress:
        from tqdm.auto import tqdm
        bar = (lambda xs, desc: tqdm(xs, desc=desc))
    per_model = {l.name: {} for l in todo}
    status = {l.name: {} for l in todo}
    for m in extractors:
        for l in bar(todo, f"trích xuất · {m}"):
            pm, st = extract_one_model(l, clients[m], workers)
            for t in pm:
                per_model[l.name].setdefault(t, {})[m] = pm[t]
                status[l.name].setdefault(t, {})[m] = st[t]
    cands = {l.name: combine_candidates(l, per_model[l.name], extractors) for l in todo}
    votes = {l.name: {} for l in todo}
    vstat = {l.name: [] for l in todo}
    for m in models:
        for l in bar(todo, f"kiểm chứng · {m}"):
            v = run_validation(cands[l.name], clients, [m], SUBJECT_NAME[l.subject], K, workers, statuses=vstat[l.name])
            for t, mv in v.items():
                votes[l.name].setdefault(t, {}).update(mv)
    for l in todo:
        calls = [x for seg in status[l.name].values() for x in seg.values()] + vstat[l.name]
        transient = any(x in TRANSIENT for x in calls)
        data = {"candidates": cands[l.name], "per_model": per_model[l.name], "status": status[l.name],
                "votes": votes[l.name], "val_status": vstat[l.name]}
        final = d / f"{l.name}.json"
        _write_atomic(d / f"{l.name}.partial.json" if transient and not allow_partial else final, data)
        if final.exists():
            (d / f"{l.name}.partial.json").unlink(missing_ok=True)
    return Path(run_dir)


def load_run(run_dir: Path) -> dict:
    files = sorted((Path(run_dir) / "lessons").glob("*.json"))
    return {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in files if not f.stem.endswith(".partial")}


def run_predictions(run: dict, models: list, K: int, alpha: float, variant: str):
    """-> (dự đoán theo bài, điểm theo bài, pool theo bài, trạng thái mọi lời gọi trích xuất + kiểm chứng)."""
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
        statuses += [s for seg in r.get("status", {}).values() for s in seg.values()] + r.get("val_status", [])
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
