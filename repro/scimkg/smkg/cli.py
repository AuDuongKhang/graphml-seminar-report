"""CLI SciMKG. Chạy từ thư mục repro/scimkg: python -m smkg.cli <lệnh> ...  (xem README.md)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def data_dir() -> Path:
    from repro.common.paths import data_root
    return data_root() / "scimkg"


def out_dir() -> Path:
    from repro.common.paths import out_root
    return out_root() / "scimkg"


def _add_download(sub):
    p = sub.add_parser("download", help="S0: tải nhãn vàng + dữ liệu thô của các bài có nhãn (Zenodo)")
    p.add_argument("--no_images", action="store_true", help="bỏ ảnh slide (~1 GB) nếu chưa cần S5")
    p.add_argument("--no_wandb", action="store_true")

    def run(a):
        import collections

        from smkg.data import download_subset, load_lessons
        d = download_subset(data_dir(), with_images=not a.no_images)
        lessons = load_lessons(d)
        stats = {"lessons": len(lessons), "segments": sum(len(l.segments) for l in lessons),
                 "gold_concepts": sum(len(l.gt) for l in lessons), "images": sum(len(l.images) for l in lessons),
                 "lessons_with_author_votes": sum(1 for l in lessons if l.author_validation),
                 **{f"lessons/{k}": v for k, v in collections.Counter(l.subject for l in lessons).items()}}
        print(stats)
        if not a.no_wandb:
            from repro.common.tracking import init_wandb
            run = init_wandb("S0-data", stats, out_dir() / "S0-data", group="S0")
            run.summary.update(stats)
            run.finish()

    p.set_defaults(func=run)


def _add_check_models(sub):
    p = sub.add_parser("check-models", help="thử gọi các mô hình; in mô hình nào dùng được / bị thay thế")
    p.add_argument("--models", nargs="+", default=["gpt-4o", "deepseek-v3", "gemini-1.5-flash", "gemini-1.5-pro",
                                                   "qwen3-14b", "gemma3-27b", "phi3-medium"])

    def run(a):
        from smkg.llm import check_models
        for k, v in check_models(a.models, out_dir() / "cache").items():
            print(f"{k:18s} {v}")

    p.set_defaults(func=run)


def _lessons(n, seed=42):
    import json

    from smkg.data import load_lessons, stratified_lessons
    manifest = json.loads((data_dir() / "manifest.json").read_text(encoding="utf-8"))
    return load_lessons(data_dir(), stratified_lessons(manifest, None if n in (None, 0) else n, seed))


def _log(exp_id, config, metrics, group, extra=None):
    from repro.common.tracking import init_wandb
    run = init_wandb(exp_id, config, out_dir() / "wandb" / exp_id, group=group)
    if extra:
        run.log(extra)
    run.summary.update(metrics)
    run.finish()


def _add_score_authors(sub):
    p = sub.add_parser("score-authors", help="S1-authors: chấm lại đầu ra LLM gốc của tác giả với nhãn vàng (không cần API)")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.5, 0.7, 0.9])

    def run(a):
        import wandb

        from smkg.runner import score_authors
        lessons = _lessons(None)
        rows = []
        for al in a.alphas:
            m = score_authors(lessons, al)
            rows.append([al] + [m[k] for k in ("precision", "recall", "f1", "acc", "mcc", "auc")])
            print(al, {k: round(v, 4) for k, v in m.items()})
        best = max(rows, key=lambda r: r[3])
        _log("S1-authors-output", {"alphas": a.alphas}, dict(zip(["alpha", "precision", "recall", "f1", "acc", "mcc", "auc"], best)),
             "S1", {"by_alpha": wandb.Table(columns=["alpha", "precision", "recall", "f1", "acc", "mcc", "auc"], data=rows)})

    p.set_defaults(func=run)


def _add_evia(sub):
    p = sub.add_parser("evia", help="S1/S3: chạy EVIA (trích xuất + self-refine) và lưu log phiếu")
    p.add_argument("--models", nargs="+", required=True, help="vd gpt-4o deepseek-v3 gemini-1.5-flash hoặc qwen3-14b gemma3-27b phi3-medium")
    p.add_argument("--extractors", nargs="+", default=None, help="mặc định = --models (bài); mã tác giả chỉ dùng deepseek-v3")
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--lessons", type=int, default=30, help="số bài (chia đều theo môn); 0 = tất cả")
    p.add_argument("--run_name", required=True, help="vd S1-paper-models hoặc S3-open-large")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--allow_partial", action="store_true",
                   help="chấp nhận bài còn lời gọi lỗi sau nhiều lần chạy lại (mặc định: để dạng .partial, chạy lại sẽ thử lại)")

    def run(a):
        from smkg.llm import make_clients
        from smkg.runner import run_evia, score_run
        lessons = _lessons(a.lessons)
        clients = make_clients(sorted(set(a.models + (a.extractors or []))), out_dir() / "cache")
        rd = run_evia(lessons, clients, a.models, a.extractors or a.models, a.K, out_dir() / "runs" / a.run_name, a.workers,
                      allow_partial=a.allow_partial)
        partial = len(list((rd / "lessons").glob("*.partial.json")))
        if partial:
            print(f"{partial} bài còn lời gọi lỗi -> chạy lại lệnh này (lời gọi đã thành công lấy từ cache)")
        m = score_run(rd, lessons, a.models, a.K, a.alpha, "full")
        m["partial_lessons"] = partial
        calls = {f"calls/{n}/{k}": v for n, c in clients.items() for k, v in c.counters.items()}
        print(m)
        resolved = {n: c.model for n, c in clients.items()}   # mô hình thực sự dùng (sau fallback nếu có)
        _log(a.run_name, {"models": a.models, "extractors": a.extractors or a.models, "K": a.K, "alpha": a.alpha,
                          "lessons": len(lessons), "resolved_models": resolved}, {**m, **calls},
             "S1" if a.run_name.startswith("S1") else "S3")

    p.set_defaults(func=run)


def _run_args(p):
    p.add_argument("--run_name", required=True)
    p.add_argument("--models", nargs="+", required=True, help="đúng thứ tự như lúc chạy evia (mô hình đầu = mô hình đơn)")
    p.add_argument("--lessons", type=int, default=30)


def _add_ablate(sub):
    p = sub.add_parser("ablate", help="S2: w/o MLs, w/o SF-FD, w/o cả hai — tính lại từ log của evia")
    _run_args(p)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.7)

    def run(a):
        from smkg.runner import score_run
        lessons = _lessons(a.lessons)
        for v in ["full", "wo_MLs", "wo_SFFD", "wo_both"]:
            m = score_run(out_dir() / "runs" / a.run_name, lessons, a.models, a.K, a.alpha, v)
            print(v, {k: round(x, 4) for k, x in m.items()})
            _log(f"S2-{a.run_name}-{v}", {"variant": v, "source_run": a.run_name, "K": a.K, "alpha": a.alpha}, m, "S2")

    p.set_defaults(func=run)


def _add_grid(sub):
    p = sub.add_parser("grid", help="S4: lưới K x alpha (Table 5) từ log của evia")
    _run_args(p)
    p.add_argument("--Ks", type=int, nargs="+", default=[1, 3, 5])
    p.add_argument("--alphas", type=float, nargs="+", default=[0.5, 0.7, 0.9])

    def run(a):
        import wandb

        from smkg.runner import score_run
        lessons = _lessons(a.lessons)
        cols = ["K", "alpha", "precision", "recall", "f1", "acc", "mcc", "auc"]
        rows = []
        for k in a.Ks:
            for al in a.alphas:
                m = score_run(out_dir() / "runs" / a.run_name, lessons, a.models, k, al, "full")
                rows.append([k, al] + [m[c] for c in cols[2:]])
                print(rows[-1])
        best = max(rows, key=lambda r: r[4])
        _log(f"S4-{a.run_name}-grid", {"source_run": a.run_name}, dict(zip(cols, best)), "S4",
             {"grid": wandb.Table(columns=cols, data=rows)})

    p.set_defaults(func=run)


def _add_errors(sub):
    p = sub.add_parser("errors", help="S5a: phân loại lỗi FE/NR/IO/CE/CO (Fig 5) từ log của evia")
    _run_args(p)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.7)

    def run(a):
        from smkg.errors import classify_errors
        from smkg.runner import load_run, run_predictions
        lessons = {l.name: l for l in _lessons(a.lessons)}
        runs = {k: v for k, v in load_run(out_dir() / "runs" / a.run_name).items() if k in lessons}
        pred, _, _, statuses = run_predictions(runs, a.models, a.K, a.alpha, "full")
        counts = classify_errors(pred, {k: lessons[k].gt for k in runs},
                                 {k: " ".join(t for _, t in lessons[k].segments) for k in runs}, statuses)
        total = sum(counts.values()) or 1
        pct = {f"pct/{k}": 100 * v / total for k, v in counts.items()}
        print(counts, pct)
        _log(f"S5-{a.run_name}-errors", {"source_run": a.run_name}, {**counts, **pct}, "S5")

    p.set_defaults(func=run)


def _add_baseline(sub):
    p = sub.add_parser("baseline", help="S1b: baseline Table 4 (decomposed_qa | gpt_ner | self_improving | universal_ner)")
    p.add_argument("--name", required=True, choices=["decomposed_qa", "gpt_ner", "self_improving", "universal_ner"])
    p.add_argument("--model", required=True, help="vd gpt-4o, qwen3-14b; universal_ner dùng uniner-7b")
    p.add_argument("--lessons", type=int, default=30)
    p.add_argument("--pool_run", default=None, help="tên run evia để lấy pool ứng viên cho ACC/MCC (khuyến nghị)")
    p.add_argument("--si_corpus", type=int, default=50, help="số đoạn không nhãn cho self_improving")

    def run(a):
        import random

        from smkg.baselines import BASELINES
        from smkg.llm import make_clients
        from smkg.metrics import set_prf, universe_metrics
        from smkg.runner import load_run, run_baseline
        lessons = _lessons(a.lessons)
        chosen = {l.name for l in lessons}
        client = make_clients([a.model], out_dir() / "cache")[a.model]
        others = [l for l in _lessons(0) if l.name not in chosen]

        def kwargs_for(l):
            if a.name != "self_improving":
                return {}
            segs = [t for o in others if o.subject == l.subject for _, t in o.segments]
            return {"corpus_segments": random.Random(42).sample(segs, min(a.si_corpus, len(segs)))}

        pred, partial = run_baseline(lessons, BASELINES[a.name], client,
                                     out_dir() / "baselines" / f"{a.name}-{a.model}", kwargs_for)
        if partial:
            print(f"{len(partial)} bài còn lời gọi lỗi -> chạy lại lệnh này để thử lại: {partial}")
        gold = {l.name: l.gt for l in lessons}
        if a.pool_run:
            run_logs = load_run(out_dir() / "runs" / a.pool_run)
            pool = {k: {c for cs in v["candidates"].values() for c in cs} for k, v in run_logs.items()}
        else:
            pool = {k: set() for k in gold}
        m = {**set_prf(pred, gold), **universe_metrics(pred, gold, pool), "n_lessons": len(gold),
             "partial_lessons": len(partial), **{f"calls/{k}": v for k, v in client.counters.items()}}
        print(m)
        _log(f"S1b-{a.name}-{a.model}", {"baseline": a.name, "model": a.model, "resolved_model": client.model,
                                          "pool_run": a.pool_run}, m, "S1")

    p.set_defaults(func=run)


def _pairs():
    import json

    from smkg.align_eval import build_image_index, load_pairs
    from smkg.data import KG_URL, _download
    kg = _download(KG_URL, data_dir() / "SciMKG.zip")
    manifest = json.loads((data_dir() / "manifest.json").read_text(encoding="utf-8"))
    pairs = load_pairs(kg, build_image_index(manifest, data_dir() / "lessons"))
    if not pairs:
        raise SystemExit("không có cặp khái niệm-ảnh: chạy `download` (không dùng --no_images) trước")
    return pairs


def _add_align_eval(sub):
    p = sub.add_parser("align-eval", help="S5b: CLIPScore / BLIP-ITM / VQAScore cho các cặp khái niệm-ảnh của SciMKG")
    p.add_argument("--metrics", nargs="+", default=["clip", "blip"], choices=["clip", "blip", "vqascore"])
    p.add_argument("--translator", default="qwen3-14b", help="LLM dịch khái niệm sang tiếng Anh")
    p.add_argument("--max_pairs", type=int, default=1000)

    def run(a):
        import random

        import wandb

        from smkg import align_eval as ae
        from smkg.llm import make_clients
        pairs = _pairs()
        pairs = random.Random(42).sample(pairs, min(a.max_pairs, len(pairs)))
        client = make_clients([a.translator], out_dir() / "cache")[a.translator]
        en = {}
        for subj in {p.subject for p in pairs}:
            en.update(ae.translate(sorted({p.concept for p in pairs if p.subject == subj}), subj, client))
        texts = [en.get(p.concept) or p.concept for p in pairs]
        cols, summary = {}, {"n_pairs": len(pairs)}
        if "clip" in a.metrics:
            cos = ae.clip_scores(pairs, texts)
            cols["clip_cos"] = cos
            summary["clip_cos_mean"] = sum(cos) / len(cos)
            summary["clipscore_mean"] = sum(ae.clipscore_from_cos(c) for c in cos) / len(cos)
        if "blip" in a.metrics:
            cols["blip_itm"] = ae.blip_itm(pairs, texts)
            summary["blip_itm_mean"] = sum(cols["blip_itm"]) / len(pairs)
        if "vqascore" in a.metrics:
            cols["vqascore"] = ae.vqascore(pairs, texts)
            summary["vqascore_mean"] = sum(cols["vqascore"]) / len(pairs)
        print(summary)
        keys = list(cols)
        table = wandb.Table(columns=["image", "concept_zh", "concept_en"] + keys,
                            data=[[wandb.Image(str(p.image)), p.concept, t] + [cols[k][i] for k in keys]
                                  for i, (p, t) in enumerate(zip(pairs[:200], texts[:200]))])
        _log("S5-align-eval", {"metrics": a.metrics, "translator": a.translator, "max_pairs": a.max_pairs}, summary,
             "S5", {"pairs": table})

    p.set_defaults(func=run)


def _add_human_sheet(sub):
    p = sub.add_parser("human-sheet", help="S5c: tạo phiếu chấm người (100 cặp) hoặc chấm phiếu đã điền (--score)")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--score", default=None, help="đường dẫn human_sheet.csv đã điền đủ 2 cột rater")

    def run(a):
        from smkg import align_eval as ae
        if a.score:
            m = ae.score_human_sheet(Path(a.score))
            print(m)
            _log("S5-human-eval", {"sheet": a.score}, m, "S5")
        else:
            print("sheet ->", ae.make_human_sheet(_pairs(), a.n, out_dir() / "human_eval"))

    p.set_defaults(func=run)


def _add_regen_align(sub):
    p = sub.add_parser("regen-align", help="S5d (tùy chọn): căn chỉnh lại ảnh bằng prompt C1-C4 và so với bản công bố")
    p.add_argument("--model", default="gemma3-27b", help="mô hình đa phương thức (vd gpt-4o nếu có key)")
    p.add_argument("--max_images", type=int, default=200)
    p.add_argument("--runs", type=int, default=3)

    def run(a):
        import collections
        import random

        from smkg.align_eval import regenerate_alignment
        from smkg.llm import make_clients
        from smkg.metrics import set_prf
        from smkg.normalize import normalize_concept
        pairs = _pairs()
        released = collections.defaultdict(set)
        subject = {}
        for p in pairs:
            released[p.image].add(normalize_concept(p.concept))
            subject[p.image] = p.subject
        lesson_concepts = {}
        for l in _lessons(0):
            lesson_concepts[l.name] = sorted(l.gt)
        images = random.Random(42).sample(sorted(released), min(a.max_images, len(released)))
        client = make_clients([a.model], out_dir() / "cache")[a.model]
        pred = {}
        for img in images:
            concepts = lesson_concepts.get(img.parent.name, [])
            pred[str(img)] = {normalize_concept(x) for x in regenerate_alignment(img, concepts, subject[img], client, a.runs)}
        m = {**set_prf(pred, {str(i): released[i] for i in images}), "n_images": len(images),
             **{f"calls/{k}": v for k, v in client.counters.items()}}
        print(m)
        _log(f"S5-regen-align-{a.model}", {"model": a.model, "runs": a.runs}, m, "S5")

    p.set_defaults(func=run)


REGISTER = [_add_download, _add_check_models, _add_score_authors, _add_evia, _add_ablate, _add_grid, _add_errors,
            _add_baseline, _add_align_eval, _add_human_sheet, _add_regen_align]


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="smkg")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for register in REGISTER:
        register(sub)
    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
