"""Kiểm thử độc lập: tất cả dữ liệu trong tệp này là tình huống giả lập, không phải mẫu tác giả."""
import json
import hashlib
from pathlib import Path
import zipfile
from types import SimpleNamespace
import pytest

from pilot import (normalize, seconds, integrate, author_score, parse_author_votes, prf,
                   read_lesson, build_graph, validate_graph, export_graph, validate_response,
                   CONCEPTS, CodexClient, extract_validate, augment, align_images,
                   explain_with_evidence, run_pilot)
import prepare_sample


def lesson():
    return {"id": "fixture", "language": "vi", "subject": "sinh học", "knowledge_point": "quang hợp",
            "source_url": "https://example.org/fixture", "video_url": "https://example.org/video",
            "duration_seconds": 120, "segments": [
                {"id": "s1", "start": "00:00", "end": "01:00", "text": "Quang hợp diễn ra ở lục lạp."},
                {"id": "s2", "start": "01:00", "end": "02:00", "text": "Lục lạp chứa diệp lục."}], "images": []}


def test_vietnamese_preserves_word_boundaries_and_diacritics():
    assert normalize("  LỤC   LẠP ") == "lục lạp"
    assert normalize("luc lap") != normalize("lục lạp")


@pytest.mark.parametrize("value,expected", [("01:02", 62), ("01:02:03", 3723)])
def test_timestamps(value, expected):
    assert seconds(value) == expected


@pytest.mark.parametrize("value", ["00:60", "01:99:00", "-1:00", "1", "00:00.5"])
def test_invalid_timestamps(value):
    with pytest.raises(ValueError): seconds(value)


def test_voting_boundary_and_failed_denominator():
    # Bảy trên mười: đạt 0,7. Thiếu một vòng phải dừng như mã chính.
    res = integrate(["a", "b"], [["a"]] * 7 + [[]] * 3, 0.7)
    assert res["concepts"] == ["a"]
    assert res["scores"] == {"a": 0.7, "b": 0.0}
    assert res["successful_rounds"] == 10 and res["failed_rounds"] == 0
    with pytest.raises(ValueError): integrate(["a"], [["a"]] * 7 + [[]] * 3 + [None])
    with pytest.raises(ValueError): integrate(["a"], [None])
    with pytest.raises(ValueError): integrate(["a"], [["new"]])


def test_author_scores_are_not_raw_counts():
    assert author_score(1) == 0 and author_score(16) == 1
    assert author_score(11) == pytest.approx(2 / 3)
    assert author_score(12) > 0.7
    with pytest.raises(ValueError): author_score(17)


def test_author_vote_parser_rejects_truncation():
    blocks = parse_author_votes("00:00-01:00\n植物 细胞\n12 1\n")
    assert blocks[0]["scores"]["细胞"] == 0
    with pytest.raises(ValueError): parse_author_votes("00:00-01:00\n植物 细胞\n12\n")


def test_author_votes_preserve_feedback_between_blocks():
    blocks = parse_author_votes('00:00-01:00\n细胞\n12\nhello, "细胞": "Phản hồi giả lập"\n1\n\n01:00-02:00\n植物\n16\n')
    assert len(blocks) == 2
    assert blocks[0]["raw_counts"] == {"细胞": 12}
    assert blocks[0]["feedback_lines"] == ['hello, "细胞": "Phản hồi giả lập"', "1"]
    assert blocks[1]["scores"] == {"植物": 1}


def test_remote_zip_reads_only_selected_ranges_and_checks_server(tmp_path):
    import io,re
    from remote_sample import RangeReader
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("chosen.txt", "Dữ liệu giả lập")
        archive.writestr("unused.bin", b"X" * 100)
    raw = buffer.getvalue()
    class Response(io.BytesIO):
        status = 206
    requests = []
    def opener(req, **kwargs):
        interval = req.headers["Range"]; requests.append(interval)
        if interval.startswith("bytes=-"):
            start, end = max(0, len(raw)-int(interval[7:])), len(raw)-1
        else:
            start, end = map(int,re.fullmatch(r"bytes=(\d+)-(\d+)",interval).groups())
        response = Response(raw[start:end+1]); response.headers = {"Content-Range": f"bytes {start}-{end}/{len(raw)}"}
        return response
    reader = RangeReader("https://example.org/fixture.zip", opener=opener)
    with zipfile.ZipFile(reader) as archive:
        assert archive.read("chosen.txt").decode() == "Dữ liệu giả lập"
    assert requests == ["bytes=-65558"]  # ZIP giả lập nằm trọn trong phần cuối.
    def whole_file(req, **kwargs):
        response = Response(raw); response.status = 200; response.headers = {}
        return response
    with pytest.raises(ValueError, match="không hỗ trợ Range"):
        RangeReader("https://example.org/fixture.zip", opener=whole_file)


def test_remote_zip_rejects_wrong_range_and_budget():
    import io
    from remote_sample import RangeReader
    class Response(io.BytesIO):
        status = 206
        headers = {"Content-Range": "bytes 0-9/100"}
    with pytest.raises(ValueError, match="phần cuối"):
        RangeReader("https://example.org/fixture.zip", opener=lambda *a,**k: Response(b"X"*10))
    with pytest.raises(ValueError, match="giới hạn"):
        RangeReader("https://example.org/fixture.zip", max_bytes=5,opener=lambda *a,**k: Response(b"X"*10))


def test_wikipedia_collector_ignores_empty_parser_container(tmp_path):
    from collect_sources import snapshot_wikipedia
    from pilot import digest
    url = "https://vi.wikipedia.org/wiki/T%E1%BA%BF_b%C3%A0o"
    target = tmp_path / (digest(url.encode())[:20] + ".html")
    target.write_text('<div class="mw-parser-output"></div><div class="mw-parser-output"><p>Tế bào là đơn vị cơ bản tạo nên các cơ thể sống trong ví dụ kiểm thử giả lập này.</p></div>')
    result = snapshot_wikipedia("tế bào", "vi", "Tế bào", tmp_path)
    assert "text" in result and "Tế bào" in result["text"]
    assert result["sha256"] == digest(target.read_bytes())


def test_metrics_by_hand():
    result = prf(["a", "b", "c"], ["a", "b", "d", "e"])
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == 0.5
    assert result["f1"] == pytest.approx(4 / 7)


def test_input_rejects_overlap_and_missing_image(tmp_path):
    data = lesson(); path = tmp_path / "lesson.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert len(read_lesson(path)["segments"]) == 2
    data["segments"][1]["start"] = "00:59"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError): read_lesson(path)
    data = lesson(); data["images"] = [{"id": "i1", "path": "missing.png"}]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError): read_lesson(path)


def test_graph_retains_both_video_occurrences_and_roundtrips(tmp_path):
    extracted = [{"segment_id": s, "concepts": ["lục lạp"]} for s in ["s1", "s2"]]
    graph = build_graph(lesson(), extracted, [], {"lục lạp": {"text": "Ví dụ & kiểm thử", "source_urls": ["https://example.org"]}}, {})
    assert len([n for n in graph["nodes"] if n["kind"] == "concept"]) == 1
    assert len([n for n in graph["nodes"] if n["kind"] == "video"]) == 2
    export_graph(graph, tmp_path)
    assert json.loads((tmp_path / "graph.json").read_text()) == graph
    broken = {"nodes": graph["nodes"], "edges": graph["edges"] + [["missing", "hasImage", "missing2"]]}
    with pytest.raises(ValueError): validate_graph(broken)


def test_external_concept_does_not_inherit_source_video():
    graph = build_graph(lesson(), [{"segment_id": "s1", "concepts": ["lục lạp"]}], [], {}, {}, {"tế bào": [{"url": "https://example.org"}]})
    extra = next(n for n in graph["nodes"] if n.get("origin") == "external")
    assert not any(a == extra["id"] and r == "hasVideo" for a, r, b in graph["edges"])


@pytest.mark.parametrize("offset,duration,valid", [(74,49,True),(100,49,False),(74,10,False)])
def test_local_video_interval_is_relative_to_clip(monkeypatch,tmp_path,offset,duration,valid):
    data=lesson();data['segments']=[{'id':'s1','start':'01:14','end':'01:49','text':'Dữ liệu giả lập'}, {'id':'s2','start':'01:49','end':'02:03','text':'Dữ liệu giả lập'}]
    data['duration_seconds']=550;data['video_path']='fixture.mp4';data['video_clip_start_seconds']=offset
    (tmp_path/'fixture.mp4').write_bytes(b'fixture-not-real-video')
    path=tmp_path/'lesson.json';path.write_text(json.dumps(data))
    monkeypatch.setattr('pilot.subprocess.run',lambda *a,**k:SimpleNamespace(stdout=json.dumps({'format':{'duration':str(duration)}})))
    if not valid:
        with pytest.raises(ValueError,match='ngoài clip'):read_lesson(path)
    else:
        parsed=read_lesson(path)
        graph=build_graph(parsed,[{'segment_id':s['id'],'concepts':['lục lạp']} for s in data['segments']],[],{}, {})
        videos=[n for n in graph['nodes'] if n['kind']=='video']
        assert [(n['local_start_seconds'],n['local_end_seconds']) for n in videos]==[(0,35),(35,49)]
        assert videos[1]['end_seconds']==123


@pytest.mark.parametrize("response", [{}, {"concepts": "a"}, {"concepts": [42]}, {"concepts": [], "extra": "x"}])
def test_malformed_outputs_fail_loudly(response):
    with pytest.raises(ValueError): validate_response(response, CONCEPTS)


def test_authentication_cannot_fall_back_to_api(monkeypatch, tmp_path):
    class Result:
        returncode = 0; stdout = "Logged in using an API key"; stderr = ""
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-secret")
    def run(argv, **kwargs):
        assert "OPENAI_API_KEY" not in kwargs["env"]
        assert 'forced_login_method="chatgpt"' in argv
        return Result()
    monkeypatch.setattr("pilot.subprocess.run", run)
    with pytest.raises(RuntimeError): CodexClient(tmp_path, "fixture-model")


def test_codex_cache_resumes_only_successful_calls(monkeypatch, tmp_path):
    calls = []
    class Result:
        returncode = 0; stdout = "Logged in using ChatGPT"; stderr = ""
    def run(argv, **kwargs):
        if "status" in argv:
            return Result()
        calls.append(argv)
        Path(argv[argv.index("-o") + 1]).write_text('{"concepts":["lục lạp"]}', encoding="utf-8")
        result = Result()
        result.stdout = '{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":3}}\n'
        return result
    monkeypatch.setattr("pilot.subprocess.run", run)
    client = CodexClient(tmp_path, "fixture-model")
    first = client.call("extraction", "fixture", CONCEPTS)
    assert client.call("extraction", "fixture", CONCEPTS) == first
    assert len(calls) == 1
    records = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert [r["cached"] for r in records] == [False, True]
    # Vòng khác phải gọi thật thay vì lấy lại cùng một phiếu từ cache.
    client.call("extraction", "fixture", CONCEPTS, round_index=1)
    assert len(calls) == 2


def test_interrupted_codex_response_is_not_cached(monkeypatch, tmp_path):
    class Result:
        returncode = 0; stdout = "Logged in using ChatGPT"; stderr = ""
    def run(argv, **kwargs):
        result = Result()
        if "status" not in argv:
            result.stdout = '{"type":"turn.failed","error":{"message":"fixture timeout"}}\n'
        return result
    monkeypatch.setattr("pilot.subprocess.run", run)
    client = CodexClient(tmp_path, "fixture-model")
    with pytest.raises(RuntimeError): client.call("extraction", "fixture", CONCEPTS)
    assert len(list(tmp_path.glob("*.failed.json"))) == 1
    assert not [p for p in tmp_path.glob("*.json") if not p.name.endswith(".failed.json")]


@pytest.mark.parametrize("language,term", [("zh", "叶绿体"), ("vi", "lục lạp")])
def test_two_language_extraction_and_refinement_without_gold(language, term):
    class Client:
        def call(self, stage, prompt, spec, **kwargs):
            assert "groundtruth" not in prompt
            return {"feedback": ""} if stage == "feedback" else {"concepts": [term]}
    data = lesson(); data["language"] = language
    if language == "zh":
        for segment in data["segments"]:
            segment["text"] = "叶绿体是植物细胞的细胞器。"
    output = extract_validate(data, Client(), rounds=1)
    assert output[0]["concepts"] == [term]
    assert output[1]["concepts"] == []  # Mã chính chỉ giữ lần xuất hiện đầu.


def test_external_augmentation_rejects_unknown_concepts():
    class Client:
        def call(self, *args, **kwargs): return {"concepts": ["bịa"]}
    with pytest.raises(ValueError):
        augment(["lục lạp"], [{"source_concept": "lục lạp", "concept": "tế bào", "relation": "PartOf", "url": "https://api.conceptnet.io/c/vi/lục_lạp"}], Client(), "vi")


def test_alignment_threshold_is_strict_and_gold_not_used():
    data = lesson(); data["images"] = [{"id": "i1", "path": "fixture.png"}]
    class Client:
        def call(self, stage, prompt, *args):
            assert "gold" not in prompt
            return {"matches": [{"concept": "lục lạp", "confidence": 0.8, "reason": "fixture"}]}
    assert align_images(data, ["lục lạp"], Client()) == []


def test_sample_archive_checksum_and_inventory(monkeypatch, tmp_path):
    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("lesson/text.json", '{"text":"dữ liệu giả lập"}')
    with pytest.raises(ValueError): prepare_sample.inspect_archive(archive)
    monkeypatch.setattr(prepare_sample, "EXPECTED_MD5", hashlib.md5(archive.read_bytes()).hexdigest())
    inventory = prepare_sample.inspect_archive(archive)
    assert inventory["sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert [f["path"] for f in inventory["files"]] == ["lesson/text.json"]


@pytest.mark.parametrize("filename", ["../outside.json", "/outside.json"])
def test_sample_archive_rejects_unsafe_paths(monkeypatch, tmp_path, filename):
    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as z: z.writestr(filename, "fixture")
    monkeypatch.setattr(prepare_sample, "EXPECTED_MD5", hashlib.md5(archive.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="không an toàn"):
        prepare_sample.inspect_archive(archive)


def test_pipeline_connects_source_external_and_modalities_with_fake_inference(tmp_path):
    data = lesson(); data["images"] = [{"id": "i1", "path": "fixture.png"}]
    class Client:
        def call(self, stage, prompt, *args, **kwargs):
            assert "groundtruth" not in prompt
            if stage == "feedback": return {"feedback": ""}
            if stage == "augmentation": return {"concepts": ["tế bào"]}
            if stage == "explanation": return {"explanation": "Giải thích giả lập chỉ dùng cho kiểm thử."}
            if stage.startswith("image-"):
                return {"matches": [{"concept": "lục lạp", "confidence": 0.9, "reason": "fixture"}]}
            return {"concepts": ["lục lạp"]}
    client = Client()
    extracted = extract_validate(data, client, rounds=1)
    added, provenance = augment(["lục lạp"], [{"source_concept": "lục lạp", "concept": "tế bào",
        "relation": "PartOf", "url": "https://example.org/fixture-conceptnet"}], client, "vi")
    concepts = ["lục lạp"] + added
    images = align_images(data, concepts, client)
    explanations, missing = explain_with_evidence(concepts, [{"concept": "lục lạp",
        "url": "https://example.org/fixture-wikipedia", "text": "Đoạn nguồn giả lập."}], client, "vi")
    assert missing == ["tế bào"]
    assert explanations["lục lạp"]["source_urls"] == ["https://example.org/fixture-wikipedia"]
    graph = build_graph(data, extracted, images, explanations,
        {"lục lạp": {"path": "fixture.aiff", "voice": "fixture", "manual_listening": "pending"}}, provenance)
    export_graph(graph, tmp_path)
    kinds = {n["kind"] for n in graph["nodes"]}
    assert {"subject", "knowledgePoint", "concept", "video", "image", "explanation", "audio"} <= kinds
    concept = next(n for n in graph["nodes"] if n["kind"] == "concept" and n["label"] == "lục lạp")
    assert {r for a, r, b in graph["edges"] if a == concept["id"]} >= {
        "hasVideo", "hasImage", "hasExplanation", "hasAudio"}


def test_failed_rerun_cannot_leave_a_successful_author_checkpoint(monkeypatch, tmp_path):
    data = lesson(); data["language"] = "zh"
    source = tmp_path / "fixture.json"; source.write_text(json.dumps(data), encoding="utf-8")
    out = tmp_path / "runs" / "fixture"; out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps({"language": "zh", "source_checked": True,
        "status": "awaiting_review"}), encoding="utf-8")
    def unavailable(*args): raise RuntimeError("fixture authentication failure")
    monkeypatch.setattr("pilot.CodexClient", unavailable)
    args = SimpleNamespace(lesson=source, out=tmp_path / "runs", model="fixture", rounds=1, alpha=0.7)
    with pytest.raises(RuntimeError): run_pilot(args)
    result = json.loads((out / "summary.json").read_text())
    assert result["status"] == "failed" and not result.get("source_checked", False)
