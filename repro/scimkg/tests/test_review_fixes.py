"""Test tái hiện các lỗi do review toàn nhánh phát hiện (I4, I5, I6, I7)."""
import json

from smkg.data import Lesson
from smkg.evia import integrate, run_validation
from smkg.llm import LLMClient, strip_think
from smkg.runner import load_run, run_evia, run_predictions


def fake(tmp_path, name, fn, cfg=None):
    c = LLMClient(name, {"provider": "fake", "model": name, **(cfg or {})}, tmp_path / name, sleep=lambda s: None)
    c._call = lambda prompt, **p: fn(prompt)
    return c


# ---------------- I4: vòng lỗi bị tính như phiếu chống ----------------
def test_integrate_ignores_failed_rounds():
    votes = {"s": {"m": [["a"], None]}}
    assert integrate({"s": ["a", "b"]}, votes, ["m"], K=2)["s"] == {"a": 1.0, "b": 0.0}
    assert integrate({"s": ["a"]}, {"s": {"m": [None, None]}}, ["m"], K=2)["s"] == {"a": 0.0}


def test_failed_refine_round_is_none_and_status_recorded(tmp_path):
    def answer(prompt):
        if "Analyze the extracted" in prompt:
            return '"b" : "x"'
        raise TimeoutError()
    statuses = []
    votes = run_validation({"s": ["a"]}, {"m": fake(tmp_path, "m", answer)}, ["m"], "Biology", K=2, workers=1,
                           statuses=statuses)
    assert votes == {"s": {"m": [None, None]}}
    assert statuses.count("timeout") == 2


LESSONS = [Lesson(name=f"L{i}", subject="biology", segments=[("00:00-01:00", f"T{i}")], images=[], gt={"a"})
           for i in range(2)]


def _ok(prompt):
    if "Analyze the extracted" in prompt:
        return '"a" : "x"'
    return "- a"


def test_lesson_with_transient_errors_is_not_finalized(tmp_path):
    flaky = {"fail": True}

    def answer(prompt):
        if flaky["fail"] and "Concept Filtering" in prompt:
            raise RuntimeError("503")
        return _ok(prompt)
    c = fake(tmp_path, "m", answer)
    run_evia(LESSONS, {"m": c}, ["m"], ["m"], 1, tmp_path / "run", workers=1, progress=False)
    assert load_run(tmp_path / "run") == {}
    flaky["fail"] = False
    run_evia(LESSONS, {"m": c}, ["m"], ["m"], 1, tmp_path / "run", workers=1, progress=False)
    run = load_run(tmp_path / "run")
    assert set(run) == {"L0", "L1"} and run["L0"]["votes"]["00:00-01:00"]["m"] == [["a"]]


def test_validation_statuses_reach_error_analysis(tmp_path):
    run_evia(LESSONS, {"m": fake(tmp_path, "m", _ok)}, ["m"], ["m"], 1, tmp_path / "run", workers=1, progress=False)
    _, _, _, statuses = run_predictions(load_run(tmp_path / "run"), ["m"], 1, 0.5, "full")
    assert len(statuses) == 2 * (1 + 2)      # mỗi bài: 1 lời gọi trích xuất + (phản hồi, sửa) x 1 vòng


# ---------------- I5: Ollama phải nạp lại mô hình liên tục ----------------
def test_run_evia_is_model_major(tmp_path):
    order = []

    def rec(name):
        def f(prompt):
            order.append(name)
            return _ok(prompt)
        return f
    clients = {"m1": fake(tmp_path, "m1", rec("m1")), "m2": fake(tmp_path, "m2", rec("m2"))}
    run_evia(LESSONS, clients, ["m1", "m2"], ["m1", "m2"], 1, tmp_path / "run", workers=1, progress=False)
    switches = sum(1 for a, b in zip(order, order[1:]) if a != b)
    assert switches <= 3          # trích xuất: m1 -> m2, rồi kiểm chứng: m1 -> m2


# ---------------- I6: Qwen3 thinking mode ----------------
def test_unclosed_think_block_counts_as_no_response():
    assert strip_think("<think>đang suy nghĩ...\n- a") == ""


def test_no_think_models_get_switch_appended(tmp_path):
    seen = []
    c = fake(tmp_path, "q", lambda p: seen.append(p) or "- a", cfg={"no_think": True})
    c.chat("P")
    assert seen == ["P /no_think"]


# ---------------- I7: tham số giải mã riêng theo mô hình ----------------
def test_per_model_params_override(tmp_path):
    seen = {}
    c = LLMClient("u", {"provider": "fake", "model": "u", "params": {"max_tokens": 256}}, tmp_path, sleep=lambda s: None)
    c._call = lambda prompt, **p: seen.update(p) or "x"
    c.chat("P")
    assert seen["max_tokens"] == 256 and seen["temperature"] == 0.0


def test_uniner_config_limits_output_tokens():
    from smkg.llm import load_model_configs
    assert load_model_configs()["uniner-7b"]["params"]["max_tokens"] <= 256
    assert load_model_configs()["qwen3-14b"].get("no_think") is True
