"""Đối chiếu độc lập phiếu, đồ thị, tài nguyên và thông tin sử dụng sau lượt chạy."""
from collections import Counter
import json
from pathlib import Path
import wave
import xml.etree.ElementTree as ET

from pilot import RDF, KG, digest, write_json


def audit_run(root, cache):
    config = json.loads((root / "config.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    if summary["status"] != "awaiting_review": raise ValueError("Lượt chưa xuất đồ thị")
    if digest((root / "input.json").read_bytes()) != config["lesson_sha256"]:
        raise ValueError("Đầu vào lưu lại khác đầu vào đã chạy")
    extracted = json.loads((root / "extraction.json").read_text())
    checks = []
    for part in extracted:
        votes = part["votes"]
        if config.get("evia_backend") and part["candidates"] and len(votes) != config["rounds"]:
            raise ValueError("EVIA của mã chính yêu cầu đủ số vòng kiểm chứng")
        counted = Counter(term for ballot in votes for term in set(ballot))
        expected = {term: counted[term] / len(votes) for term in part["candidates"]}
        selected = [term for term, value in expected.items() if value >= config["alpha"]]
        if expected != part["scores"] or selected != part["concepts"]:
            raise ValueError("Tính phiếu độc lập khác đầu ra")
        checks.append({"segment_id": part["segment_id"], "votes": len(votes),
                       "scores_equal": True, "selection_equal": True})
    graph = json.loads((root / "graph.json").read_text())
    doc = ET.parse(root / "graph.rdf").getroot()
    rdf_ids = {el.attrib["{" + RDF + "}about"].removeprefix(KG) for el in doc}
    if rdf_ids != {n["id"] for n in graph["nodes"]}: raise ValueError("RDF mất nút")
    rdf_labels = {el.attrib["{" + RDF + "}about"].removeprefix(KG): el.find("{" + KG + "}label").text for el in doc}
    if rdf_labels != {n["id"]: n["label"] for n in graph["nodes"]}: raise ValueError("RDF đổi nhãn")
    rdf_edges = set()
    for el in doc:
        for child in el:
            if child.tag.startswith("{" + KG + "}") and "{" + RDF + "}resource" in child.attrib:
                rdf_edges.add((el.attrib["{" + RDF + "}about"].removeprefix(KG), child.tag.removeprefix("{" + KG + "}"), child.attrib["{" + RDF + "}resource"].removeprefix(KG)))
    if rdf_edges != {tuple(e) for e in graph["edges"]}: raise ValueError("RDF đổi cạnh")
    audio = []
    videos = []
    input_data = json.loads((root / "input.json").read_text())
    for node in graph["nodes"]:
        if node["kind"] == "video" and node.get("local_path"):
            if digest(Path(node["local_path"]).read_bytes()) != node["clip_sha256"]:
                raise ValueError("Video đổi nội dung")
            offset = input_data.get("video_clip_start_seconds", 0)
            if (node["local_start_seconds"] != node["start_seconds"] - offset
                    or node["local_end_seconds"] != node["end_seconds"] - offset
                    or node["local_start_seconds"] < 0
                    or node["local_end_seconds"] > node["clip_duration_seconds"] + 0.15):
                raise ValueError("Ánh xạ mốc thời gian clip không đúng")
            videos.append({"id":node["id"],"sha256":node["clip_sha256"],"source_interval":[node["start_seconds"],node["end_seconds"]],
                           "local_interval":[node["local_start_seconds"],node["local_end_seconds"]],"manual_content_review":"pending"})
        if node["kind"] == "audio":
            path = Path(node["path"])
            if digest(path.read_bytes()) != node["sha256"]: raise ValueError("Âm thanh đổi nội dung")
            with wave.open(str(path)) as stream:
                audio.append({"id": node["id"], "frames": stream.getnframes(), "sample_rate": stream.getframerate(),
                              "duration_seconds": stream.getnframes()/stream.getframerate(), "listening": "pending"})
    logs = [json.loads(line) for line in (cache / "calls.jsonl").read_text().splitlines()]
    calls = [x for x in logs if x.get("run_id") == summary["lesson_id"]]
    actual, reused, failures = [x for x in calls if not x["cached"] and "error" not in x], [x for x in calls if x["cached"]], [x for x in calls if "error" in x]
    totals = {key: sum(x.get("usage", {}).get(key, 0) for x in actual) for key in ["input_tokens", "output_tokens", "cached_input_tokens"]}
    stages = {}
    for name in dict.fromkeys(x["stage"] for x in actual):
        rows = [x for x in actual if x["stage"] == name]
        stages[name] = {"calls": len(rows), "elapsed_seconds": sum(x["elapsed_seconds"] for x in rows),
                        **{key: sum(x.get("usage", {}).get(key, 0) for x in rows) for key in totals}}
    all_keys = list(dict.fromkeys(x["key"] for x in calls if "error" not in x))
    for key in all_keys:
        record = json.loads((cache / (key + ".json")).read_text())
        if record["authentication"] != "chatgpt": raise ValueError("Có lời gọi ngoài thuê bao")
        if any(marker in record["prompt"] for marker in ["groundtruth", "annotation_provenance", "vietnamese-gold.json", "author-gold.json"]):
            raise ValueError("Prompt chứa dấu vết tệp nhãn đánh giá")
        for event in record["events"]:
            if event.get("item", {}).get("type") in ["command_execution", "web_search", "mcp_tool_call"]:
                raise ValueError("Có lượt dùng công cụ trong phép đo")
    report = {"lesson_id": summary["lesson_id"], "language": summary["language"],
              "input_sha256": config["lesson_sha256"], "input_snapshot_equal": True,
              "gold_file_markers_absent_in_prompts": True,
              "voting": checks, "rdf_nodes_equal": True, "rdf_labels_equal": True, "rdf_edges_equal": True,
              "node_kinds": dict(Counter(x["kind"] for x in graph["nodes"])), "audio": audio, "local_video":videos,
              "usage": {"actual_calls": len(actual), "local_cache_hits": len(reused), "failed_calls": len(failures),
                        "unique_keys": len(all_keys), "elapsed_inference_seconds": sum(x["elapsed_seconds"] for x in actual),
                        **totals, "by_stage": stages, "api_cost_usd": None, "authentication": "chatgpt",
                        "measurement": "Tổng lời gọi thật trong nhật ký của mẫu, gồm các bước chạy tách và chạy ghép; token CLI gồm ngữ cảnh hệ thống."},
              "single_extraction_query": next((x for x in actual if x["stage"] == "extraction"), None),
              "manual_reviews": "pending"}
    write_json(root / "audit.json", report)
    return report


def check_main_evia():
    """Replay two frozen samples and smoke-check wiring; never invoke Codex."""
    from contextlib import redirect_stdout
    import inspect
    import io
    import re
    import tempfile
    from types import SimpleNamespace
    from unittest.mock import patch
    import pilot
    from main_evia import backend, fingerprint, extract_validate, augment_from_main

    root = Path(__file__).resolve().parent
    source = backend()
    call_lines = {name: inspect.getsourcelines(getattr(source, name))[1]
                  for name in ("Extraction", "Validation", "SELF_REFINE", "Intergration", "expandExplanations")}
    snapshots = list((root / "runs/cache").glob("*.json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in snapshots
               if not path.name.endswith(".failed.json")]
    references = {
        "author-mitochondria": ("data/author-lesson.json", "data/author-conceptnet.json", "data/author-sources/wikipedia.json"),
        "vi-photosynthesis": ("data/vietnamese-lesson.json", "data/vietnamese-conceptnet.json", "data/vi-sources/wikipedia.json"),
    }
    cases = []
    for lesson_id, (lesson_path, candidates_path, evidence_path) in references.items():
        lesson = pilot.read_lesson(root / lesson_path)
        old = json.loads((root / "runs" / lesson_id / "extraction.json").read_text(encoding="utf-8"))
        old_explanations = json.loads((root / "runs" / lesson_id / "explanations.json").read_text(encoding="utf-8"))
        old_by_id = {part["segment_id"]: part for part in old}
        segments = {seg["id"]: seg for seg in lesson["segments"]}
        candidates = json.loads((root / candidates_path).read_text(encoding="utf-8"))
        evidence = json.loads((root / evidence_path).read_text(encoding="utf-8"))

        class ReplayClient:
            model = "gpt-6.1-sol"

            def __init__(self):
                self.calls = []

            def call(self, stage, prompt, spec, *args, **kwargs):
                self.calls.append(stage)
                if stage in ("extraction", "refinement", "feedback"):
                    segment_id = re.search(r"^Đoạn: (.+)$", prompt, re.MULTILINE).group(1)
                    if stage == "extraction":
                        return {"concepts": old_by_id[segment_id]["candidates"]}
                    if stage == "refinement":
                        return {"concepts": old_by_id[segment_id]["votes"][kwargs["round_index"]]}
                    matched = [record for record in records if record.get("stage") == "feedback"
                               and record.get("round") == kwargs["round_index"]
                               and segments[segment_id]["text"] in record["prompt"]]
                    if len(matched) != 1:
                        raise ValueError("Phản hồi lưu đệm không xác định duy nhất")
                    return matched[0]["response"]
                if stage == "explanation":
                    term = re.search(r"^Khái niệm: (.+)$", prompt, re.MULTILINE).group(1)
                    return {"explanation": old_explanations[term]["text"]}
                if stage.startswith("image-"):
                    return {"matches": []}  # Wiring smoke only; no new image assessment.
                raise ValueError("Không có phản hồi lưu cho bước " + stage)

        replay = ReplayClient()
        seen, expected = set(), {}
        for part in old:
            pool = sorted(term for term in part["candidates"] if term not in seen)
            seen.update(part["candidates"])
            expected[part["segment_id"]] = {
                "candidates": pool,
                "scores": {term: sum(term in ballot for ballot in part["votes"]) / len(part["votes"]) for term in pool},
            }
        current = extract_validate(lesson, replay, rounds=5, alpha=0.7)
        for part in current:
            wanted = expected[part["segment_id"]]
            assert part["candidates"] == wanted["candidates"]
            assert part["scores"] == wanted["scores"]
            assert part["concepts"] == [term for term in wanted["candidates"] if wanted["scores"][term] >= 0.7]
            assert part["failed_rounds"] == 0
        concepts = pilot.unique(term for part in current for term in part["concepts"])
        combinations = []
        for conceptnet, wikipedia in ((False, False), (True, False), (False, True), (True, True)):
            result = augment_from_main(lesson, concepts, candidates, evidence, replay, conceptnet, wikipedia)
            additions, provenance, explanations, missing, status = result
            assert not additions  # Main ConceptNet extraction rules are still empty.
            assert bool(explanations) == (conceptnet or wikipedia)
            if not (conceptnet or wikipedia):
                assert status == {"conceptnet": "disabled", "wikipedia": "disabled"}
            else:
                assert bool(status["snapshot_lookups"]["conceptnet"]) == conceptnet
                assert bool(status["snapshot_lookups"]["wikipedia"]) == wikipedia
                assert all(bool(item["source_urls"]) == item["grounded"] for item in explanations.values())
            # Real runner wiring; only image answers are synthetic and TTS is off.
            with tempfile.TemporaryDirectory(prefix="scimkg-main-runner-") as temporary:
                target = Path(temporary)
                checkpoint = target / "author.json"
                checkpoint.write_text(json.dumps({"language": "zh", "source_checked": True, "status": "awaiting_review"}))
                args = SimpleNamespace(lesson=root / lesson_path, out=target, model=replay.model,
                                       rounds=5, alpha=0.7, author_checkpoint=checkpoint,
                                       candidates=root / candidates_path, evidence=root / evidence_path,
                                       use_conceptnet=conceptnet, use_wikipedia=wikipedia, speech=False, gold=None)
                with patch.object(pilot, "CodexClient", lambda *unused: ReplayClient()), redirect_stdout(io.StringIO()):
                    pilot.run_pilot(args)
                run = target / lesson_id
                summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
                assert summary["status"] == "awaiting_review" and summary["external_resources"] == status
                assert json.loads((run / "explanations.json").read_text(encoding="utf-8")) == explanations
                graph = json.loads((run / "graph.json").read_text(encoding="utf-8"))
                pilot.validate_graph(graph)
                assert (run / "graph.rdf").is_file()
                # A changed flag cannot overwrite a recorded run.
                config_before = (run / "config.json").read_bytes()
                args.use_wikipedia = not wikipedia
                try:
                    pilot.run_pilot(args)
                except ValueError as exc:
                    assert "cấu hình khác" in str(exc)
                else:
                    raise AssertionError("Cấu hình khác không được ghi đè")
                assert (run / "config.json").read_bytes() == config_before
            combinations.append({"use_conceptnet": conceptnet, "use_wikipedia": wikipedia,
                                 "additions": len(additions), "explanations": len(explanations),
                                 "missing_grounding": len(missing), "status": status,
                                 "json_rdf_runner_smoke": True, "config_overwrite_rejected": True, "passed": True})
        fallback = augment_from_main(lesson, concepts, [], [], replay, False, True)
        assert set(fallback[3]) == set(concepts)
        assert all(not item["grounded"] and not item["source_urls"] for item in fallback[2].values())

        frozen_files = [lesson_path, candidates_path, evidence_path,
                        f"runs/{lesson_id}/extraction.json", f"runs/{lesson_id}/explanations.json"]
        cases.append({"lesson_id": lesson_id, "language": lesson["language"], "segments": len(current),
                      "repeated_candidates_removed": sum(len(part["candidates"]) for part in old)
                                                     - sum(len(part["candidates"]) for part in current),
                      "selected_unique_concepts": len(concepts), "voting_matches_main": True,
                      "resource_combinations": combinations, "fallback_marked_unverified": True,
                      "json_rdf_runner_smoke": True, "image_smoke_response": "synthetic_empty",
                      "speech_requested": False,
                      "recorded_inputs_sha256": {path: digest((root / path).read_bytes()) for path in frozen_files}})
    try:
        pilot.integrate(["x"], [["x"], None])
    except ValueError:
        failed_round_rejected = True
    else:
        raise AssertionError("Vòng thiếu không được chấp nhận")
    assert source.parse_concept_list("- lục lạp\n- tế bào thực vật\n") == ["lục lạp", "tế bào thực vật"]
    assert pilot.integrate(["below", "at"], [["below", "at"]] * 6 + [["at"]] + [[]] * 3)["concepts"] == ["at"]
    assert pilot.integrate(["zero"], [[]], alpha=0)["concepts"] == ["zero"]
    main_defaults = source.parse_arguments(["unused", "--subject", "Biology"])
    assert not main_defaults.use_conceptnet and not main_defaults.use_wikipedia
    failures_propagated = []
    for failure_stage in ("extraction", "feedback", "refinement"):
        class FailingClient:
            def call(self, stage, *args, **kwargs):
                if stage == failure_stage:
                    raise RuntimeError("offline failure: " + stage)
                return {"feedback": ""} if stage == "feedback" else {"concepts": ["lục lạp"]}
        try:
            extract_validate(lesson, FailingClient(), rounds=1)
        except RuntimeError as exc:
            assert str(exc) == "offline failure: " + failure_stage
            failures_propagated.append(failure_stage)
        else:
            raise AssertionError("Lỗi mô hình không được coi là dự đoán rỗng")
    votes_path = root / "data/author-votes.json"
    votes = json.loads(votes_path.read_text(encoding="utf-8"))
    rows = [[part["interval"], list(part["raw_counts"]), list(part["raw_counts"].values()), "",
             [0] * len(part["raw_counts"])] for part in votes]
    with tempfile.TemporaryDirectory(prefix="scimkg-author-votes-") as temporary, redirect_stdout(io.StringIO()):
        selected = source.Intergration(rows, 5, str(Path(temporary) / "lesson.txt"), model_count=3)
    assert set(term for _, terms in selected for term in terms) == {"有氧呼吸", "酶", "能量供应"}
    report = {
        "date": "2026-10-11", "source": fingerprint(), "source_function_lines": call_lines,
        "runner_sha256": digest((root / "pilot.py").read_bytes()),
        "audit_runner_sha256": digest(Path(__file__).read_bytes()),
        "scope": "Recorded-response replay and structural smoke; not fresh model predictions or live resource APIs",
        "actual_llm_calls": 0, "paid_api_calls": 0, "same_model_configuration_as_paper": False,
        "pilot_validation_models": 1, "failed_round_rejected": failed_round_rejected,
        "llm_failure_stages_propagated": failures_propagated, "resource_defaults_match_main": True,
        "recorded_call_log_sha256": digest((root / "runs/cache/calls.jsonl").read_bytes()),
        "threshold_boundary": {"alpha": 0.7, "votes": 7, "total": 10, "accepted": True},
        "alpha_zero_matches_main": True, "vietnamese_full_terms_preserved": True,
        "author_first": True, "samples": cases,
        "author_vote_replay": {"source_sha256": digest(votes_path.read_bytes()),
                              "models": 3, "rounds": 5, "selected": ["有氧呼吸", "酶", "能量供应"],
                              "published_integration_selected_count": 1, "generating_config_known": False},
        "remaining": ["ConceptNet extraction rules are empty",
                      "Live Wikipedia proxy/translation configuration remains unverified",
                      "Main legacy text cache splits multiword terms on reload; pilot uses fresh temporary files and JSON",
                      "One Codex model with JSON/language adaptation and frozen snapshots"],
    }
    write_json(root / "results/main-evia-check.json", report)
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, nargs="?")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--check-main-evia", action="store_true", help="Đối chiếu EVIA với fork bằng phản hồi đã lưu; không gọi LLM")
    args = parser.parse_args()
    if args.check_main_evia:
        result = check_main_evia()
    else:
        if not args.run or not args.cache:
            parser.error("Cần run và --cache, hoặc --check-main-evia")
        result = audit_run(args.run, args.cache)
    print(json.dumps(result, ensure_ascii=False, indent=2))
