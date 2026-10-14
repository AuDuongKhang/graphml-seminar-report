"""Pilot dùng EVIA của fork SciMKG; Codex thuê bao và nguồn chụp được ghi riêng."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import unicodedata
import wave
import xml.etree.ElementTree as ET

RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
KG = "https://example.org/scimkg-pilot/"
INVERSE = {"relatedTo": "relatedTo", "hasExplanation": "isExplanationOf",
           "hasImage": "isImageOf", "hasVideo": "isVideoOf", "hasAudio": "isAudioOf"}


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize(term: str) -> str:
    # Giữ khoảng cách giữa từ và dấu tiếng Việt; không dùng cách bỏ mọi khoảng trắng của tiếng Trung.
    return " ".join(unicodedata.normalize("NFKC", term).strip().split()).casefold()


def unique(terms):
    return list(dict.fromkeys(normalize(x) for x in terms if normalize(x)))


def seconds(timestamp: str) -> int:
    if not re.fullmatch(r"\d{1,3}:\d{2}(?::\d{2})?", timestamp):
        raise ValueError("Mốc thời gian phải có dạng mm:ss hoặc hh:mm:ss")
    parts = [int(x) for x in timestamp.split(":")]
    if any(x >= 60 for x in parts[1:]):
        raise ValueError("Phút/giây ngoài khoảng hợp lệ")
    return sum(x * 60 ** i for i, x in enumerate(reversed(parts)))


def read_lesson(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    for field in ("id", "language", "subject", "knowledge_point", "source_url", "segments"):
        if not data.get(field):
            raise ValueError(f"Thiếu trường {field}")
    if data["language"] not in ("zh", "vi"):
        raise ValueError("Chỉ hỗ trợ mẫu tiếng Trung và bài tiếng Việt trong lần chạy thử")
    seen, end = set(), 0
    for seg in data["segments"]:
        if seg["id"] in seen or not seg["text"].strip():
            raise ValueError("Đoạn trùng định danh hoặc thiếu văn bản")
        seen.add(seg["id"])
        start, stop = seconds(seg["start"]), seconds(seg["end"])
        if start < end or stop <= start or stop > data["duration_seconds"]:
            raise ValueError("Đoạn chồng nhau hoặc vượt thời lượng nguồn")
        end = stop
    for image in data.get("images", []):
        if image["id"] in seen:
            raise ValueError("Tài nguyên trùng định danh")
        seen.add(image["id"])
        target = (path.parent / image["path"]).resolve()
        if not target.is_file():
            raise ValueError(f"Thiếu ảnh {image['id']}")
        image["path"] = str(target)
    if data.get("video_path"):
        target = (path.parent / data["video_path"]).resolve()
        if not target.is_file():
            raise ValueError("Thiếu tệp video")
        data["video_path"] = str(target)
        start = data.get("video_clip_start_seconds", 0)
        if isinstance(start, bool) or not isinstance(start, (int, float)) or start < 0:
            raise ValueError("Mốc đầu clip video không hợp lệ")
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(target)],
                               capture_output=True, text=True, check=True, timeout=20)
        duration = float(json.loads(probe.stdout)["format"]["duration"])
        if not duration > 0 or any(seconds(s["start"]) < start or seconds(s["end"]) > start + duration + 0.15 for s in data["segments"]):
            raise ValueError("Đoạn nằm ngoài clip video cục bộ")
        data["video_clip_duration_seconds"] = duration
        data["video_sha256"] = digest(target.read_bytes())
    return data


def integrate(candidates, rounds, alpha=0.7):
    from main_evia import backend
    if not 0 <= alpha <= 1:
        raise ValueError("Ngưỡng phải thuộc [0, 1]")
    if not rounds or any(r is None for r in rounds):
        raise ValueError("Kiểm chứng chưa hoàn tất; dừng thay vì đổi mẫu số")
    available = [set(unique(r)) for r in rounds]
    pool = unique(candidates)
    if any(not r <= set(pool) for r in available):
        raise ValueError("Kiểm chứng đã thêm khái niệm ngoài tập ứng viên")
    scores = {c: sum(c in r for r in available) / len(available) for c in pool}
    candidate = [["pilot", pool, [1 + sum(c in r for r in available) for c in pool], "", [0] * len(pool)]]
    with tempfile.TemporaryDirectory(prefix="scimkg-integration-") as temporary, redirect_stdout(io.StringIO()):
        accepted = backend().Intergration(candidate, len(rounds), str(Path(temporary) / "lesson.txt"),
                                         model_count=1, alpha=alpha)[0][1]
    return {"scores": scores, "concepts": [c for c in pool if c in accepted],
            "successful_rounds": len(available), "failed_rounds": 0}


def author_score(raw: int, validators=3, rounds=5) -> float:
    total = validators * rounds
    if validators <= 0 or rounds <= 0 or not isinstance(raw, int) or not 1 <= raw <= 1 + total:
        raise ValueError("Điểm gốc hoặc cấu hình ngoài khoảng hợp lệ")
    return (raw - 1) / total


def prf(predicted, gold):
    p, g = set(unique(predicted)), set(unique(gold))
    tp = len(p & g)
    precision, recall = tp / len(p) if p else 0, tp / len(g) if g else 0
    return {"precision": precision, "recall": recall,
            "f1": 2 * tp / (len(p) + len(g)) if p or g else 0,
            "tp": tp, "predicted": len(p), "gold": len(g),
            "false_positive": sorted(p - g), "missed": sorted(g - p)}


def parse_author_votes(text: str):
    """Giữ cả phần phản hồi của phiếu thật; chuẩn hóa dòng điểm ngay sau ứng viên."""
    rows = [line.strip() for line in text.lstrip("\ufeff").splitlines() if line.strip()]
    result = []
    i = 0
    while i < len(rows):
        if not re.fullmatch(r"\d{2}:\d{2}-\d{2}:\d{2}", rows[i]):
            raise ValueError("Khối phiếu thiếu mốc thời gian; không âm thầm bỏ dữ liệu")
        if i + 2 >= len(rows):
            raise ValueError("Khối phiếu chưa đầy đủ")
        terms, counts = rows[i + 1].split(), rows[i + 2].split()
        if len(terms) != len(counts) or any(not x.isdigit() for x in counts):
            raise ValueError("Khái niệm và phiếu không khớp")
        interval = rows[i]
        start, end = interval.split("-")
        if seconds(end) <= seconds(start):
            raise ValueError("Khoảng phiếu không hợp lệ")
        if any(x["interval"] == interval for x in result):
            raise ValueError("Khối phiếu trùng khoảng")
        j = i + 3
        while j < len(rows) and not re.fullmatch(r"\d{2}:\d{2}-\d{2}:\d{2}", rows[j]):
            j += 1
        result.append({"interval": interval, "scores": {t: author_score(int(c)) for t, c in zip(terms, counts)},
                       "raw_counts": dict(zip(terms, map(int, counts))), "feedback_lines": rows[i + 3:j]})
        i = j
    if not result:
        raise ValueError("Tệp phiếu rỗng")
    return result


def schema(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


CONCEPTS = schema({"concepts": {"type": "array", "items": {"type": "string"}}})
FEEDBACK = schema({"feedback": {"type": "string"}})
EXPLANATION = schema({"explanation": {"type": "string"}})
ALIGNMENT = schema({"matches": {"type": "array", "items": schema({
    "concept": {"type": "string"}, "confidence": {"type": "number"}, "reason": {"type": "string"}})}})


def validate_response(value, spec):
    kind = spec["type"]
    if kind == "object":
        if not isinstance(value, dict) or set(value) != set(spec["properties"]):
            raise ValueError("Phản hồi sai trường JSON")
        for key, child in spec["properties"].items():
            validate_response(value[key], child)
    elif kind == "array":
        if not isinstance(value, list):
            raise ValueError("Phản hồi không phải danh sách")
        for item in value:
            validate_response(item, spec["items"])
    elif kind == "string" and not isinstance(value, str):
        raise ValueError("Phản hồi không phải chuỗi")
    elif kind == "number" and (isinstance(value, bool) or not isinstance(value, (int, float))):
        raise ValueError("Phản hồi không phải số")


class CodexClient:
    """Chỉ đăng nhập ChatGPT, không dùng API key; cache đầu ra thành công theo nội dung."""
    def __init__(self, root: Path, model: str, run_id=None):
        self.root, self.model = root, model
        self.run_id = run_id
        self.env = {k: v for k, v in os.environ.items() if k not in ("OPENAI_API_KEY", "CODEX_API_KEY", "CODEX_ACCESS_TOKEN")}
        status = subprocess.run(["codex", "-c", 'forced_login_method="chatgpt"', "login", "status"],
                                capture_output=True, text=True, env=self.env, timeout=20)
        if status.returncode or "Logged in using ChatGPT" not in status.stdout + status.stderr:
            raise RuntimeError("Codex chưa đăng nhập bằng ChatGPT; dừng, không chuyển sang API")
        self.root.mkdir(parents=True, exist_ok=True)

    def call(self, stage, prompt, spec, image=None, round_index=None):
        payload = {"model": self.model, "stage": stage, "prompt": prompt, "schema": spec,
                   "round": round_index, "image_hash": digest(Path(image).read_bytes()) if image else None}
        key = digest(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode())
        record = self.root / (key + ".json")
        if record.exists():
            saved = json.loads(record.read_text(encoding="utf-8"))
            validate_response(saved["response"], spec)
            self.log({"stage": stage, "key": key, "cached": True, "elapsed_seconds": 0})
            return saved["response"]
        with tempfile.TemporaryDirectory(prefix="scimkg-codex-") as tmp:
            folder = Path(tmp)
            write_json(folder / "schema.json", spec)
            argv = ["codex", "exec", "--ignore-user-config", "--ephemeral", "--skip-git-repo-check",
                    "--sandbox", "read-only", "--json", "-m", self.model,
                    "-c", 'forced_login_method="chatgpt"', "-c", 'model_provider="openai"',
                    "-c", 'model_reasoning_effort="low"', "--output-schema", str(folder / "schema.json"),
                    "-o", str(folder / "reply.json"), "-C", tmp]
            for feature in ("shell_tool", "multi_agent", "apps", "plugins", "browser_use", "computer_use", "skill_search", "image_generation"):
                argv += ["--disable", feature]
            if image:
                argv += ["--image", str(Path(image).resolve())]
            argv += ["-"]
            start = time.monotonic()
            instruction = "Chỉ thực hiện tác vụ suy luận dưới đây và trả JSON. Không dùng công cụ, không đọc tệp, không tìm mạng. "
            try:
                run = subprocess.run(argv, input=instruction + prompt, capture_output=True, text=True,
                                     env=self.env, timeout=180)
                # Chỉ là sự kiện của lời gọi; không đọc hoặc sao chép kho thông tin đăng nhập.
                (self.root / (key + ".events.jsonl")).write_text(run.stdout, encoding="utf-8")
                (self.root / (key + ".stderr.txt")).write_text(run.stderr, encoding="utf-8")
                events = [json.loads(line) for line in run.stdout.splitlines() if line.strip().startswith("{")]
                completed = [e for e in events if e.get("type") == "turn.completed"]
                if run.returncode or not completed:
                    raise RuntimeError("Codex chưa hoàn tất lượt; xem nhật ký lỗi, không có API dự phòng")
                if any(e.get("item", {}).get("type") in ("command_execution", "web_search", "mcp_tool_call") for e in events):
                    raise RuntimeError("Lượt đã dùng công cụ; không chấp nhận cho phép đo suy luận riêng")
                response = json.loads((folder / "reply.json").read_text(encoding="utf-8"))
                validate_response(response, spec)
                saved = {**payload, "response": response, "usage": completed[-1].get("usage"),
                         "events": events, "elapsed_seconds": time.monotonic() - start,
                         "authentication": "chatgpt", "api_cost_usd": None}
                write_json(record, saved)
                self.log({"stage": stage, "key": key, "cached": False, "elapsed_seconds": saved["elapsed_seconds"],
                          "usage": saved["usage"], "authentication": "chatgpt", "api_cost_usd": None})
                return response
            except Exception as exc:
                write_json(self.root / (key + ".failed.json"), {**payload, "error": str(exc),
                           "elapsed_seconds": time.monotonic() - start})
                self.log({"stage": stage, "key": key, "cached": False, "error": str(exc)})
                raise

    def log(self, value):
        if self.run_id:
            value = {"run_id": self.run_id, **value}
        with (self.root / "calls.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(value, ensure_ascii=False) + "\n")


def extract_validate(lesson, client, rounds=5, alpha=0.7):
    from main_evia import extract_validate as run_main_evia
    return run_main_evia(lesson, client, rounds, alpha)


def align_images(lesson, concepts, client):
    from main_pipeline import activate
    activate()
    from MultimodalAlignment.ImageAligment import align_images as main_align
    return main_align(lesson, concepts, client)


def augment(concepts, candidates, client, language, subject="khoa học tự nhiên"):
    """Quy tắc pilot lịch sử; run_pilot hiện gọi augment_from_main."""
    allowed_relations = {"IsA", "RelatedTo", "HasContext", "PartOf", "DefinedAs", "Synonym", "DerivedFrom", "HasPrerequisite"}
    source_set, pool, provenance = set(concepts), [], {}
    counts = {}
    for item in candidates:
        source, term = normalize(item["source_concept"]), normalize(item["concept"])
        if source not in source_set or item["relation"] not in allowed_relations or not item.get("url"):
            raise ValueError("Ứng viên bổ sung thiếu nguồn hoặc quan hệ hợp lệ")
        if term and term not in source_set and counts.get(source, 0) < 2:
            counts[source] = counts.get(source, 0) + 1
            pool.append(term)
            provenance.setdefault(term, []).append(item)
    if not pool:
        return [], {}
    answer = client.call("augmentation", f"Môn: {subject}. Ngôn ngữ: {language}. Giữ khái niệm liên quan môn, không đổi tên và không thêm ngoài ứng viên. Nguồn ConceptNet:\n" + json.dumps(candidates, ensure_ascii=False) + "\nỨng viên được xét: " + json.dumps(unique(pool), ensure_ascii=False), CONCEPTS)
    kept = unique(answer["concepts"])
    if not set(kept) <= set(pool):
        raise ValueError("Bổ sung sinh khái niệm ngoài nguồn")
    return kept, {term: provenance[term] for term in kept}


def explain_with_evidence(concepts, evidence, client, language):
    """Quy tắc giải thích lịch sử; không dùng trong run_pilot hiện tại."""
    out, missing = {}, []
    for term in concepts:
        sources = [x for x in evidence if normalize(x["concept"]) == term and x.get("url") and x.get("text", "").strip()]
        if not sources:
            missing.append(term)
            continue
        prompt = f"Viết giải thích ngắn cho {term} bằng {'tiếng Việt' if language == 'vi' else 'tiếng Trung giản thể'}. Chỉ dựa trên các trích đoạn nguồn sau; không dùng kiến thức ngoài nguồn.\n" + json.dumps(sources, ensure_ascii=False)
        response = client.call("explanation", prompt, EXPLANATION)
        if not response["explanation"].strip():
            raise ValueError("Giải thích rỗng")
        out[term] = {"text": response["explanation"], "source_urls": [x["url"] for x in sources]}
    return out, missing


def speak(explanations, root, language):
    from main_pipeline import activate
    activate()
    from MultimodalAlignment.SpeechGenerator import speak as main_speak
    return main_speak(explanations, root, language)


def build_graph(lesson, extracted, image_links, explanations, audio, additions=None):
    from main_pipeline import activate
    activate()
    from SciMKGConstruction.GeneratorSubjectJson import build_graph as main_build
    return main_build(lesson, extracted, image_links, explanations, audio, additions)


def validate_graph(graph):
    from main_pipeline import activate
    activate()
    from SciMKGConstruction.GeneratorTriples import validate_graph as main_validate
    main_validate(graph)
    return {"nodes": len(graph["nodes"]), "edges": len(graph["edges"])}


def export_graph(graph, root):
    from main_pipeline import activate
    activate()
    from RDF.rdf import export_graph as main_export
    write_json(root / "graph.json", graph)
    return main_export(graph, root)


def run_pilot(args):
    from main_pipeline import run
    return run(args)


def main():
    parser = argparse.ArgumentParser(description="Chạy thử EVIA của fork SciMKG; mẫu tác giả trước, tiếng Việt sau")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check-input", help="Kiểm tra đầu vào, không gọi mô hình")
    check.add_argument("lesson", type=Path)
    votes = sub.add_parser("check-author-votes", help="Đọc và chuẩn hóa phiếu của một mẫu tác giả")
    votes.add_argument("file", type=Path)
    run = sub.add_parser("run", help="Chạy một mẫu bằng thuê bao Codex; dừng nếu không có đăng nhập ChatGPT")
    run.add_argument("lesson", type=Path)
    run.add_argument("--out", type=Path, default=Path(__file__).parent / "runs" / "main-source")
    run.add_argument("--model", default="gpt-6.1-sol")
    run.add_argument("--rounds", type=int, default=5)
    run.add_argument("--alpha", type=float, default=0.7)
    run.add_argument("--offline", action="store_true", help="Chỉ đọc cache, không gọi Codex mới")
    run.add_argument("--max-calls", type=int, default=100)
    run.add_argument("--gold", type=Path)
    run.add_argument("--evidence", type=Path)
    run.add_argument("--candidates", type=Path, help="Ảnh chụp dữ liệu ConceptNet có URL nguồn, chỉ cho một mẫu")
    run.add_argument("--use-conceptnet", action="store_true", help="Bật ConceptNet; mặc định tắt")
    run.add_argument("--use-wikipedia", action="store_true", help="Bật Wikipedia; mặc định tắt")
    run.add_argument("--speech", action="store_true")
    run.add_argument("--author-checkpoint", type=Path)
    args = parser.parse_args()
    if args.command == "check-input":
        lesson = read_lesson(args.lesson)
        print(json.dumps({"valid": True, "segments": len(lesson["segments"]), "images": len(lesson.get("images", []))}))
    elif args.command == "check-author-votes":
        print(json.dumps(parse_author_votes(args.file.read_text(encoding="utf-8")), ensure_ascii=False, indent=2))
    else:
        run_pilot(args)


if __name__ == "__main__":
    main()
