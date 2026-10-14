"""Adapt Codex JSON and frozen resource snapshots to the fork's EVIA functions."""
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types

SOURCE = Path(__file__).resolve().parents[3] / "SciMKG" / "ExtractMultimodalConcepts"


def fingerprint():
    files = {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
             for name in ("NER.py", "ConceptNet.py", "Wikipediaapi_model.py")}
    git = subprocess.run(["git", "-C", str(SOURCE), "describe", "--always", "--dirty"],
                         capture_output=True, text=True, timeout=10)
    return {"source": "SciMKG/ExtractMultimodalConcepts/NER.py",
            "revision": git.stdout.strip() if git.returncode == 0 else None,
            "sha256": files, "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def backend():
    if not (SOURCE / "NER.py").is_file():
        raise RuntimeError("Thiếu fork SciMKG; khởi tạo submodule trước khi chạy pilot")
    key = hashlib.sha256((SOURCE / "NER.py").read_bytes()).hexdigest()[:16]
    package_name = "_scimkg_pilot_" + key
    name = package_name + ".NER"
    if name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(SOURCE)]
        sys.modules[package_name] = package
        spec = importlib.util.spec_from_file_location(name, SOURCE / "NER.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


class JsonAdapter:
    def __init__(self, lesson, client):
        self.lesson, self.client = lesson, client
        self.active = None
        self.extraction_index = 0
        self.round_index = 0
        self.raw_extraction, self.votes, self.raw_refinement = {}, {}, {}
        self.pool = []

    def __call__(self, requested_model, prompt):
        from pilot import CONCEPTS, FEEDBACK, EXPLANATION, unique

        if "Identify all nouns/noun phrases" in prompt:
            stage, spec = "extraction", CONCEPTS
            self.active = self.lesson["segments"][self.extraction_index]["id"]
            self.extraction_index += 1
        elif "Concept Filtering Protocol" in prompt:
            stage, spec = "refinement", CONCEPTS
        elif "Analyze the extracted" in prompt:
            stage, spec = "feedback", FEEDBACK
        elif "K-12 pure" in prompt:
            stage, spec = "augmentation", CONCEPTS
        else:
            stage, spec = "explanation", EXPLANATION
        language = "tiếng Việt" if self.lesson["language"] == "vi" else "tiếng Trung giản thể"
        if self.lesson["language"] == "vi":
            prompt = prompt.replace("In Simplified Chinese", "In Vietnamese")
            prompt = prompt.replace("Chinese sentence", "Vietnamese sentence")
            prompt = prompt.replace("Chinese description", "Vietnamese description")
        marker = f"Đoạn: {self.active}\n" if stage in ("extraction", "feedback", "refinement") else ""
        if stage == "explanation":
            terms = sorted(backend().finalEntity, key=len, reverse=True)
            term = next((term for term in terms if prompt.endswith(term)), None)
            if term is None:
                raise ValueError("Không xác định được khái niệm trong prompt giải thích của mã chính")
            marker = f"Khái niệm: {term}\n"
        prompt = marker + f"Ngôn ngữ đầu ra: {language}.\n" + prompt
        prompt += "\nTrả JSON đúng schema; danh sách Markdown sẽ do bộ chuyển đổi tạo."
        kwargs = {"round_index": self.round_index} if stage in ("feedback", "refinement") else {}
        response = self.client.call(stage, prompt, spec, **kwargs)
        if stage == "feedback":
            return response["feedback"]
        if stage == "explanation":
            if not response["explanation"].strip():
                raise ValueError("Giải thích rỗng")
            return response["explanation"]
        terms = unique(response["concepts"])
        if stage == "extraction":
            self.raw_extraction[self.active] = terms
        elif stage == "refinement":
            self.raw_refinement.setdefault(self.active, []).append(terms)
            self.votes.setdefault(self.active, []).append([term for term in terms if term in self.pool])
            self.round_index += 1
        return "\n".join("- " + term for term in terms)


def extract_validate(lesson, client, rounds=5, alpha=0.7):
    from pilot import author_score

    if not 1 <= rounds <= 5 or not 0 <= alpha <= 1:
        raise ValueError("Pilot yêu cầu 1–5 vòng và ngưỡng trong [0,1]")
    source = backend()
    source.allEntity.clear()
    source.finalEntity.clear()
    adapter = JsonAdapter(lesson, client)
    original_ner = source.ner
    source.ner = adapter
    models = [getattr(client, "model", "pilot-model")]
    by_time = {f"{seg['start']}-{seg['end']}": seg for seg in lesson["segments"]}
    try:
        with tempfile.TemporaryDirectory(prefix="scimkg-main-evia-") as temporary, redirect_stdout(io.StringIO()):
            path = str(Path(temporary) / "lesson.txt")
            text = "\n".join(json.dumps(interval, ensure_ascii=False) + ": " +
                             json.dumps(seg["text"], ensure_ascii=False) for interval, seg in by_time.items())
            Path(path).write_text(text + "\n", encoding="utf-8")
            candidates = source.Extraction(path, models, lesson["subject"], source.allEntity, rounds)
            if len(candidates) != len(lesson["segments"]):
                raise ValueError("Mã chính không trích xuất đủ các đoạn")
            validated = []
            for row in candidates:
                adapter.active = by_time[row[0]]["id"]
                adapter.round_index, adapter.pool = 0, list(row[1])
                validated.extend(source.Validation([row], models, lesson["subject"], rounds, path))
            integrated = dict(source.Intergration(validated, rounds, path, model_count=1, alpha=alpha))
            results = []
            for row in candidates:
                segment_id, pool = by_time[row[0]]["id"], list(row[1])
                votes = adapter.votes.get(segment_id, [])
                if pool and len(votes) != rounds:
                    raise ValueError("Kiểm chứng chưa hoàn tất đủ số vòng")
                scores = {term: author_score(raw, validators=1, rounds=rounds)
                          for term, raw in zip(pool, row[2])}
                results.append({
                    "segment_id": segment_id, "candidates": pool,
                    "raw_extraction": adapter.raw_extraction[segment_id],
                    "votes": votes, "refinement_responses": adapter.raw_refinement.get(segment_id, []),
                    "raw_scores": dict(zip(pool, row[2])), "scores": scores,
                    "concepts": [term for term in pool if term in integrated.get(row[0], set())],
                    "successful_rounds": len(votes), "failed_rounds": 0,
                })
            return results
    finally:
        source.ner = original_ner


def augment_from_main(lesson, concepts, candidates, evidence, client,
                      use_conceptnet=False, use_wikipedia=False):
    from pilot import normalize

    if not (use_conceptnet or use_wikipedia):
        return [], {}, {}, [], {"conceptnet": "disabled", "wikipedia": "disabled"}
    source = backend()
    source.finalEntity.clear()
    source.finalEntity.update(concepts)
    source.allEntity.update(concepts)
    calls = {"conceptnet": 0, "wikipedia": 0}
    allowed = {"IsA", "RelatedTo", "HasContext", "PartOf", "DefinedAs",
               "Synonym", "DerivedFrom", "HasPrerequisite"}
    for item in candidates if use_conceptnet else []:
        if (normalize(item["source_concept"]) not in concepts or item["relation"] not in allowed
                or not item.get("url")):
            raise ValueError("Ảnh chụp ConceptNet thiếu nguồn hoặc quan hệ hợp lệ")
    sources = {term: [item for item in evidence if normalize(item["concept"]) == term
                     and item.get("url") and item.get("text", "").strip()] for term in concepts}

    def conceptnet_lookup(term):
        calls["conceptnet"] += 1
        records = [item for item in candidates if normalize(item["source_concept"]) == term]
        # Preserve the main parser: its empty rules currently yield no additions.
        return ([json.dumps(item, ensure_ascii=False) for item in records], term) if records else (None, None)

    def wikipedia_lookup(term):
        calls["wikipedia"] += 1
        return {term: "\n\n".join(item["text"] for item in sources.get(term, [])) or "not found"}

    original_ner = source.ner
    source.ner = JsonAdapter(lesson, client)
    try:
        with redirect_stdout(io.StringIO()):
            pairs = source.expandExplanations(
                lesson["subject"], use_conceptnet=use_conceptnet, use_wikipedia=use_wikipedia,
                conceptnet_lookup=conceptnet_lookup, wikipedia_lookup=wikipedia_lookup)
    finally:
        source.ner = original_ner
    additions = [term for term in sorted(source.finalEntity) if term not in concepts]
    provenance = {term: [item for item in candidates if normalize(item["concept"]) == term] for term in additions}
    explanations, missing = {}, []
    for term, text in pairs:
        items = sources.get(term, []) if use_wikipedia else []
        if not items:
            missing.append(term)
        explanations[term] = {
            "text": text, "source_urls": list(dict.fromkeys(item["url"] for item in items)),
            "grounded": bool(items), "manual_review": "pending",
            "grounding_status": "snapshot_available" if items else "llm_fallback_unverified",
        }
    status = {
        "conceptnet": "completed_empty_rules" if use_conceptnet else "disabled",
        "wikipedia": "completed_with_missing_evidence" if use_wikipedia and missing else
                     "completed" if use_wikipedia else "disabled",
        "snapshot_lookups": calls,
    }
    return additions, provenance, explanations, missing, status
