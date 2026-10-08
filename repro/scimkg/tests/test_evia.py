from smkg.data import Lesson
from smkg.evia import (ablation_scores, author_scores, integrate, run_extraction, run_validation, select)
from smkg.llm import LLMClient


def fake(tmp_path, name, fn):
    c = LLMClient(name, {"provider": "fake", "model": name}, tmp_path / name, sleep=lambda s: None)
    c._call = lambda prompt, **p: fn(prompt)
    return c


def test_integrate_normalizes_votes():
    cands = {"s1": ["上皮组织", "系统"]}
    votes = {"s1": {"m1": [["上皮组织"], ["上皮组织"]], "m2": [["上皮组织", "系统"], ["上皮组织"]]}}
    out = integrate(cands, votes, ["m1", "m2"], K=2)
    assert out["s1"]["上皮组织"] == 1.0
    assert abs(out["s1"]["系统"] - 0.25) < 1e-9
    assert select(out, 0.7) == {"s1": {"上皮组织"}}


def test_integrate_prefix_rounds_for_table5():
    cands = {"s1": ["a"]}
    votes = {"s1": {"m": [["a"], [], [], [], []]}}
    assert integrate(cands, votes, ["m"], K=1)["s1"]["a"] == 1.0
    assert integrate(cands, votes, ["m"], K=5)["s1"]["a"] == 0.2


def test_run_extraction_assigns_concept_to_first_segment(tmp_path):
    lesson = Lesson(name="L", subject="biology", segments=[("00:00-01:00", "AAA"), ("01:00-02:00", "BBB")],
                    images=[], gt=set())
    reply = {"AAA": "-a\n-b", "BBB": "-b\n-c"}
    clients = {"m1": fake(tmp_path, "m1", lambda p: reply[p[:3]]), "m2": fake(tmp_path, "m2", lambda p: reply[p[:3]])}
    ext = run_extraction(lesson, clients, ["m1", "m2"], workers=1)
    assert ext["candidates"] == {"00:00-01:00": ["a", "b"], "01:00-02:00": ["c"]}
    assert ext["per_model"]["00:00-01:00"]["m1"] == ["a", "b"]


def test_run_validation_logs_survivors_per_round(tmp_path):
    def answer(prompt):
        return '"b" : "ambiguous"' if "Analyze the extracted" in prompt else "- a"
    clients = {"m": fake(tmp_path, "m", answer)}
    votes = run_validation({"s": ["a", "b"]}, clients, ["m"], "Biology", K=3, workers=1)
    assert votes == {"s": {"m": [["a"], ["a"], ["a"]]}}


def test_ablations_reuse_logs():
    per_model = {"s": {"m1": ["a", "b"], "m2": ["a", "c"]}}
    cands = {"s": ["a", "b", "c"]}
    votes = {"s": {"m1": [["a", "b"], ["a"]], "m2": [["a", "c"], ["c"]]}}
    full = ablation_scores("full", cands, per_model, votes, ["m1", "m2"], K=2)
    assert full["s"] == {"a": 0.75, "b": 0.25, "c": 0.5}
    wo_ml = ablation_scores("wo_MLs", cands, per_model, votes, ["m1", "m2"], K=2)
    assert wo_ml["s"] == {"a": 1.0, "b": 0.5}                       # chỉ ứng viên + phiếu của m1
    wo_sf = ablation_scores("wo_SFFD", cands, per_model, votes, ["m1", "m2"], K=2)
    assert wo_sf["s"] == {"a": 1.0, "b": 0.5, "c": 0.5}             # tỉ lệ LLM trích ra
    wo_both = ablation_scores("wo_both", cands, per_model, votes, ["m1", "m2"], K=2)
    assert wo_both["s"] == {"a": 1.0, "b": 1.0}


def test_author_scores_normalize_votes():
    lesson = Lesson(name="L", subject="biology", segments=[], images=[], gt=set(),
                    author_validation={"t": {"肺": 11, "环境": 1, "鳃": 16}})
    s = author_scores(lesson)
    assert s["t"]["鳃"] == 1.0 and s["t"]["环境"] == 0.0 and abs(s["t"]["肺"] - 10 / 15) < 1e-9


def test_ablations_use_the_run_extractors_when_they_differ_from_validators():
    # evia --models m1 m2 --extractors x : chỉ x trích xuất
    per_model = {"s": {"x": ["a", "b"]}}
    cands = {"s": ["a", "b"]}
    votes = {"s": {"m1": [["a"]], "m2": [["a", "b"]]}}
    assert ablation_scores("wo_SFFD", cands, per_model, votes, ["m1", "m2"], K=1)["s"] == {"a": 1.0, "b": 1.0}
    assert ablation_scores("wo_both", cands, per_model, votes, ["m1", "m2"], K=1)["s"] == {"a": 1.0, "b": 1.0}
    assert ablation_scores("wo_MLs", cands, per_model, votes, ["m1", "m2"], K=1)["s"] == {"a": 1.0, "b": 0.0}
