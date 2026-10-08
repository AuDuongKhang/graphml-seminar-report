import csv
import json
import zipfile
from pathlib import Path

from smkg.align_eval import (Pair, build_image_index, clipscore_from_cos, load_pairs, make_human_sheet,
                             parse_translations, score_human_sheet)
from smkg.llm import LLMClient


def test_build_image_index_first_occurrence_wins(tmp_path):
    a = tmp_path / "L1" / "5.png"
    b = tmp_path / "L2" / "5.png"
    for p in (a, b):
        p.parent.mkdir(parents=True)
        p.write_bytes(b"x")
    idx = build_image_index({"L1": "biology", "L2": "biology"}, tmp_path)
    assert idx[("biology", "5.png")] == a


def test_load_pairs_from_kg_zip(tmp_path):
    z = tmp_path / "SciMKG.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("biology/bioloy.json", json.dumps([
            {"id": 0, "concept": "肽链", "png": ["5.png", "9.png"]}, {"id": 1, "concept": "细胞", "png": ["none"]}]))
    img = tmp_path / "5.png"
    img.write_bytes(b"x")
    pairs = load_pairs(z, {("biology", "5.png"): img})
    assert pairs == [Pair(concept="肽链", image=img, subject="biology")]


def test_parse_translations_keeps_order_and_length():
    assert parse_translations("- peptide chain\n- cell\n", 2) == ["peptide chain", "cell"]
    assert parse_translations("- peptide chain", 2) == ["peptide chain", ""]


def test_clipscore_from_cos():
    assert clipscore_from_cos(0.3) == 0.75 and clipscore_from_cos(-0.1) == 0.0


def _pairs(tmp_path, n):
    out = []
    for i in range(n):
        p = tmp_path / f"{i}.png"
        p.write_bytes(b"x")
        out.append(Pair(concept=f"c{i}", image=p, subject="biology"))
    return out


def test_make_human_sheet_is_stable(tmp_path):
    pairs = _pairs(tmp_path, 10)
    f1 = make_human_sheet(pairs, 4, tmp_path / "s1", seed=1)
    f2 = make_human_sheet(pairs, 4, tmp_path / "s2", seed=1)
    rows1 = list(csv.DictReader(open(f1, encoding="utf-8")))
    rows2 = list(csv.DictReader(open(f2, encoding="utf-8")))
    assert len(rows1) == 4 and [r["concept_zh"] for r in rows1] == [r["concept_zh"] for r in rows2]
    assert set(rows1[0]) >= {"pair_id", "concept_zh", "concept_en", "image_file", "rater_1", "rater_2"}
    assert all((Path(f1).parent / "images" / r["image_file"]).exists() for r in rows1)


def test_score_human_sheet(tmp_path):
    f = tmp_path / "s.csv"
    f.write_text("pair_id,concept_zh,concept_en,image_file,rater_1,rater_2\n"
                 "0,a,,0.png,1,1\n1,b,,1.png,0,0\n2,c,,2.png,1,0\n3,d,,3.png,0.5,1\n", encoding="utf-8")
    m = score_human_sheet(f)
    assert abs(m["mean"] - (1 + 1 + 0 + 0 + 1 + 0 + 0.5 + 1) / 8) < 1e-9
    assert m["n"] == 4 and -1 <= m["cohen_kappa"] <= 1


def test_client_sends_image_as_data_url(tmp_path):
    c = LLMClient("v", {"provider": "ollama", "model": "v"}, tmp_path, sleep=lambda s: None)
    seen = {}

    class Fake:
        class chat:
            class completions:
                @staticmethod
                def create(model, messages, **p):
                    seen["content"] = messages[0]["content"]
                    msg = type("M", (), {"content": "- 细胞"})()
                    return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()
    c._client = Fake()
    img = tmp_path / "x.png"
    img.write_bytes(b"\x89PNG")
    assert c.chat("P", image=img).text == "- 细胞"
    kinds = [part["type"] for part in seen["content"]]
    assert kinds == ["text", "image_url"] and seen["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    c._client = None
    assert c.chat("P", image=img).cached          # khóa cache gồm cả ảnh
