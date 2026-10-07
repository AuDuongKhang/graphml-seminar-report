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

    def run(a):
        from smkg.llm import make_clients
        from smkg.runner import run_evia, score_run
        lessons = _lessons(a.lessons)
        clients = make_clients(sorted(set(a.models + (a.extractors or []))), out_dir() / "cache")
        rd = run_evia(lessons, clients, a.models, a.extractors or a.models, a.K, out_dir() / "runs" / a.run_name, a.workers)
        m = score_run(rd, lessons, a.models, a.K, a.alpha, "full")
        calls = {f"calls/{n}/{k}": v for n, c in clients.items() for k, v in c.counters.items()}
        print(m)
        _log(a.run_name, {"models": a.models, "extractors": a.extractors or a.models, "K": a.K, "alpha": a.alpha,
                          "lessons": len(lessons)}, {**m, **calls}, "S1" if a.run_name.startswith("S1") else "S3")

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


REGISTER = [_add_download, _add_check_models, _add_score_authors, _add_evia, _add_ablate, _add_grid, _add_errors]


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
