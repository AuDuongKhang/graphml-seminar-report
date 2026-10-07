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


REGISTER = [_add_download, _add_check_models]


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
