"""Pilot chỉ chuyển tham số: toàn bộ luồng xây dựng chạy trong fork SciMKG."""
from pathlib import Path
import json
import sys

SOURCE = Path(__file__).resolve().parents[3] / "SciMKG"


def activate():
    if not (SOURCE / "pipeline.py").is_file():
        raise RuntimeError("Thiếu mã chính SciMKG/pipeline.py")
    if str(SOURCE) not in sys.path:
        sys.path.insert(0, str(SOURCE))


def run(args):
    activate()
    from pipeline import run as run_main
    lesson = json.loads(args.lesson.read_text(encoding="utf-8"))
    if lesson["language"] == "vi":
        if not args.author_checkpoint or not args.author_checkpoint.is_file():
            raise ValueError("Cần lượt mẫu tác giả hoàn tất trước bài Việt")
        checkpoint = json.loads(args.author_checkpoint.read_text(encoding="utf-8"))
        configuration = checkpoint.get("configuration", {})
        author_input = Path(configuration.get("input", ""))
        if not author_input.is_file() or json.loads(author_input.read_text(encoding="utf-8")).get("language") != "zh":
            raise ValueError("Checkpoint không thuộc mẫu tiếng Trung")
        if checkpoint.get("quality_review") is None or not checkpoint.get("rdf"):
            raise ValueError("Checkpoint tác giả chưa hoàn tất luồng chính")
    if args.gold:
        raise ValueError("Lượt kiểm tra mã chưa chấm chất lượng; Ex01–Ex02 sẽ dùng nhãn công bố đúng toàn bài")
    output = args.out / lesson["id"]
    if (output / "config.json").is_file():
        raise ValueError("Đây là lượt pilot cũ; chọn --out mới để giữ lịch sử")
    summary = run_main(args.lesson, output, model=args.model, rounds=args.rounds, alpha=args.alpha,
                       use_conceptnet=args.use_conceptnet, use_wikipedia=args.use_wikipedia,
                       candidates_path=args.candidates, evidence_path=args.evidence, speech=args.speech,
                       offline=args.offline, max_calls=args.max_calls)
    print(json.dumps({key: summary[key] for key in ("lesson_id", "new_calls", "cached_calls", "nodes", "edges")},
                     ensure_ascii=False))
    return summary
