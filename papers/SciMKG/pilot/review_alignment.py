"""Đối chiếu trực tiếp mã chính với hai cặp ảnh–khái niệm trong Example.zip."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parents[2] / "SciMKG"
sys.path.insert(0, str(SOURCE))
from llm import CodexClient
from MultimodalAlignment.ImageAligment import align_images
from MultimodalAlignment.SpeechGenerator import speak
from runtime import digest, read_json, read_lesson, write_json
from SciMKGConstruction.GeneratorSubjectJson import build_graph


def run(offline=False, run_name="alignment-author-review-v3"):
    if run_name != Path(run_name).name or not run_name:
        raise ValueError("Tên lượt phải là tên thư mục đơn")
    review_file = ROOT / "results" / (run_name + ".json")
    previous_review = read_json(review_file) if review_file.exists() else {}
    baseline = read_json(ROOT / "results/alignment-author-review.json")
    output = ROOT / "runs/main-source" / run_name
    output.mkdir(parents=True, exist_ok=True)
    examples = ROOT / "data/example"
    source_hashes = {name: digest((SOURCE / name).read_bytes()) for name in (
        "MultimodalAlignment/ImageAligment.py", "llm.py", "runtime.py",
        "MultimodalAlignment/SpeechGenerator.py", "SciMKGConstruction/GeneratorSubjectJson.py")}
    if previous_review and previous_review["source_sha256"] != source_hashes:
        raise ValueError("Mã đã đổi; chọn --run-name mới để giữ bằng chứng lượt trước")
    for name in source_hashes:
        snapshot = output / "source" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE / name, snapshot)
    cases, reference = [], []
    with zipfile.ZipFile(ROOT / "data/Example.zip") as archive:
        for number, label in enumerate(("Mitochondria", "Kidney"), 1):
            source_image = examples / label / "image.jpg"
            source_text = examples / label / "text.txt"
            for path in (source_image, source_text):
                member = path.relative_to(examples).as_posix()
                if archive.read(member) != path.read_bytes():
                    raise ValueError(f"Mẫu khác thành phần gốc trong ZIP: {member}")
            # Tên tệp và ID không tiết lộ nhãn đúng cho mô hình.
            image = output / "inputs" / f"image-{number:02}.jpg"
            image.parent.mkdir(exist_ok=True)
            shutil.copyfile(source_image, image)
            lesson = {"id": f"example-{number:02}", "language": "en", "subject": "Biology",
                      "knowledge_point": "Biology example",
                      "source_url": "https://zenodo.org/records/19691725",
                      "segments": [{"id": "s1", "text": source_text.read_text().strip()}],
                      "images": [{"id": f"image-{number:02}", "path": str(image.resolve()),
                                  "sha256": digest(image.read_bytes())}]}
            cases.append({"id": f"example-{number:02}", "lesson": lesson,
                          "concepts": ["mitochondria", "kidney"]})
            reference.append({"id": f"example-{number:02}", "author_label": label.casefold(),
                              "author_label_vi": "ti thể" if number == 1 else "thận",
                              "must_include": label.casefold(),
                              "must_exclude": "kidney" if number == 1 else "mitochondria",
                              "positive_basis": f"Example.zip/{label}/image.jpg",
                              "negative_basis": "Đối chứng đổi nhãn do nhóm thiết kế; không phải nhãn âm công bố.",
                              "source_image": str(source_image.relative_to(ROOT)),
                              "source_image_sha256": digest(source_image.read_bytes()),
                              "source_text_sha256": digest(source_text.read_bytes())})
    # Lưu tiêu chí trước suy luận; chúng không nằm trong prompt gửi mô hình.
    write_json(output / "reference.json", reference)
    original = read_lesson(ROOT / "data/author-lesson-corrected.json")
    original["images"] = original["images"][:1]
    parts = read_json(ROOT / "runs/main-source/author-mitochondria/extracted.json")
    terms = list(dict.fromkeys(term for part in parts for term in part["concepts"]))
    cases.append({"id": "previous-image-12", "lesson": original, "concepts": terms})
    write_json(output / "cases.json", cases)
    client = CodexClient(output / "calls", max_calls=6, offline=offline)
    results = []
    for case in cases:
        links = align_images(case["lesson"], case["concepts"], client)
        predicted = sorted({match["concept"] for match in links})
        prior = next(row for row in baseline["image_cases"] if row["case_id"] == case["id"])
        item = {"case_id": case["id"], "links": links, "predicted": predicted,
                "baseline_links": prior["links"]}
        last_call = json.loads((client.root / "calls.jsonl").read_text().splitlines()[-1])
        item["verification"] = read_json(client.root / (last_call["key"] + ".json"))["response"]
        if case["id"].startswith("example-"):
            expected = next(row for row in reference if row["id"] == case["id"])
            item.update(author_label=expected["author_label"],
                        author_label_vi=expected["author_label_vi"],
                        positive_matches_author=expected["must_include"] in predicted,
                        rejects_swapped_label=expected["must_exclude"] not in predicted)
        else:
            item.update(expert_gold_available=False,
                        previous_links=read_json(ROOT / "runs/main-source/author-mitochondria/image-links.json"),
                        note="Ảnh 12.png chưa có nhãn chuẩn ảnh–khái niệm; kiểm tra độ ổn định, không chấm đúng/sai bằng suy đoán.")
        results.append(item)
        write_json(output / "image-results.json", results)
        print(json.dumps(item, ensure_ascii=False), flush=True)

    # Video: dùng đúng URL và khoảng thời gian tác giả công bố cho mẫu Example,
    # không ghép chúng vào bài RawData khác.
    lesson = cases[0]["lesson"]
    text = lesson["segments"][0]["text"]
    video_reference = (examples / "Mitochondria/video.txt").read_text()
    time_range = re.search(r'"time":\s*"([^"]+)"', video_reference).group(1)
    url = re.search(r'"videoUrl":\s*"([^"]+)"', video_reference).group(1)
    start, end = time_range.split("-")
    lesson["video_url"] = url
    lesson["segments"][0].update(start=start, end=end)
    explanations = {"mitochondria": {"text": text, "grounded": False,
                                     "source_urls": [], "provenance": "author-example-text"}}
    write_json(output / "tts-input.json", explanations)
    audio = speak(explanations, output / "audio", "en")
    graph = build_graph(lesson, [{"segment_id": "s1", "concepts": ["mitochondria"]}],
                        results[0]["links"], explanations, audio)
    write_json(output / "graph.json", graph)
    write_json(output / "audio.json", audio)
    video = next(node for node in graph["nodes"] if node["kind"] == "video")
    image_node = next(node for node in graph["nodes"] if node["kind"] == "image")
    concept = next(node for node in graph["nodes"] if node["kind"] == "concept")
    audio_node = next(node for node in graph["nodes"] if node["kind"] == "audio")
    explanation = next(node for node in graph["nodes"] if node["kind"] == "explanation")
    assert video["start_seconds"] == 82 and video["end_seconds"] == 132 and video["url"] == url
    for kind, resource in (("Image", image_node), ("Video", video), ("Audio", audio_node), ("Explanation", explanation)):
        assert [concept["id"], "has" + kind, resource["id"]] in graph["edges"]
        assert [resource["id"], "is" + kind + "Of", concept["id"]] in graph["edges"]
    assert explanation["text"] == (examples / "Mitochondria/text.txt").read_text().strip()
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "json", str(examples / "Mitochondria/Mitochondria.mp3")],
                           capture_output=True, text=True, check=True)
    modalities = {"video_matches_published_metadata": True, "video_start_seconds": 82,
                  "video_end_seconds": 132, "video_content_verified": False,
                  "tts_input_matches_author_explanation": True,
                  "audio_resource_links_valid": True,
                  "generated_wav_duration_seconds": audio["mitochondria"]["duration_seconds"],
                  "author_mp3_duration_seconds": float(json.loads(probe.stdout)["format"]["duration"]),
                  "audio_quality_scored": False,
                  "tts_adaptation": "macOS Samantha thay cho TTS tác giả; không yêu cầu sóng âm giống nhau."}
    logs = [json.loads(line) for line in (output / "calls/calls.jsonl").read_text().splitlines()]
    live = [item for item in logs if not item.get("cached")]
    usage = {key: sum(item.get("usage", {}).get(key, 0) for item in live) for key in (
        "input_tokens", "output_tokens", "cached_input_tokens")}
    summary = {"date": datetime.now(timezone.utc).isoformat(), "source_sha256": source_hashes,
               "model": client.model, "authentication": client.authentication, "threshold": "> 0.8",
               "new_calls_this_execution": client.new_calls, "recorded_live_calls": len(live),
               "recorded_usage": usage, "image_resource_links_valid": True,
               "api_cost_usd": None, "image_cases": results, "other_modalities": modalities,
               "limitations": ["Chỉ hai cặp ảnh–khái niệm công bố trong Example.zip.",
                               "Những cặp này là đầu ra tham chiếu của tác giả, không phải bộ nhãn chuyên gia đầy đủ.",
                               "Đối chứng đổi nhãn do nhóm tạo; không phải nhãn âm của tác giả.",
                               "Đề xuất và kiểm chứng dùng cùng Codex; khác mô hình cấu hình bài báo.",
                               "Ảnh 12.png chưa có nhãn chuẩn; video chỉ đối chiếu siêu dữ liệu, âm thanh chưa chấm nghe."]}
    # Nhận xét sau chạy không phải dự đoán và không được mất khi phát lại cache.
    summary["baseline_results_file"] = "results/alignment-author-review.json"
    summary["review_findings"] = previous_review.get("review_findings", baseline.get("review_findings", []))
    for field in ("diagnostic_results_file", "presentation_vi", "review_comparison", "controlled_checks"):
        if field in previous_review:
            summary[field] = previous_review[field]
    write_json(output / "summary.json", summary)
    write_json(review_file, summary)
    print(json.dumps({"new_calls": client.new_calls, "output": str(output),
                      "other_modalities": modalities}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--run-name", default="alignment-author-review-v3")
    args = parser.parse_args()
    run(args.offline, args.run_name)
