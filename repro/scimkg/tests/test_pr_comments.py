"""Test tái hiện 2 nhận xét trên PR #1 (fallback mô hình; baseline lưu kết quả sau lời gọi lỗi)."""
from smkg.data import Lesson
from smkg.llm import LLMClient
from smkg.runner import run_baseline


# ---------------- PR #1, llm.py:153 — fallback chỉ dùng khi check-models ----------------
def test_client_switches_to_fallback_when_model_unavailable(tmp_path):
    c = LLMClient("gemini-1.5-flash", {"provider": "gemini", "model": "gemini-1.5-flash", "fallback": "gemini-2.0-flash"},
                  tmp_path, sleep=lambda s: None)

    def call(prompt, **p):
        if c.model == "gemini-1.5-flash":
            raise RuntimeError("404 model not found")
        return "- a"
    c._call = call
    assert c.chat("P").text == "- a"
    assert c.model == "gemini-2.0-flash" and c.counters["fallback_used"] == 1
    assert c.chat("Q").text == "- a" and c.counters["fallback_used"] == 1   # các lời gọi sau dùng luôn mô hình thay thế


def test_client_without_fallback_still_reports_error(tmp_path):
    c = LLMClient("m", {"provider": "x", "model": "m"}, tmp_path, sleep=lambda s: None)
    c._call = lambda prompt, **p: (_ for _ in ()).throw(RuntimeError("404"))
    r = c.chat("P")
    assert r.text is None and r.error == "api_error" and c.model == "m"


# ---------------- PR #1, cli.py:229 — baseline lỗi tạm thời bị lưu như đã xong ----------------
LESSONS = [Lesson(name="L0", subject="biology", segments=[("t", "T")], images=[], gt={"a"})]


def _baseline(lesson, client, subject):
    r = client.chat("extract")
    return {"t": ["a"] if r.text else []}


def test_baseline_with_failed_calls_is_retried(tmp_path):
    flaky = {"fail": True}
    c = LLMClient("m", {"provider": "x", "model": "m"}, tmp_path / "cache", sleep=lambda s: None)

    def call(prompt, **p):
        if flaky["fail"]:
            raise TimeoutError()
        return "- a"
    c._call = call
    pred, partial = run_baseline(LESSONS, _baseline, c, tmp_path / "out")
    assert partial == ["L0"] and not (tmp_path / "out" / "L0.json").exists()
    flaky["fail"] = False
    pred, partial = run_baseline(LESSONS, _baseline, c, tmp_path / "out")
    assert partial == [] and pred == {"L0": {"a"}} and (tmp_path / "out" / "L0.json").exists()
    assert not (tmp_path / "out" / "L0.partial.json").exists()


def test_baseline_finished_lessons_are_not_recomputed(tmp_path):
    calls = []
    c = LLMClient("m", {"provider": "x", "model": "m"}, tmp_path / "cache", sleep=lambda s: None)
    c._call = lambda prompt, **p: calls.append(1) or "- a"
    run_baseline(LESSONS, _baseline, c, tmp_path / "out")
    run_baseline(LESSONS, _baseline, c, tmp_path / "out")
    assert len(calls) == 1


# ---------------- PR #1, llm.py:164 — check-models báo "ok" dù đã chuyển sang fallback ----------------
def test_check_models_reports_fallback_not_ok(tmp_path, monkeypatch):
    from smkg import llm

    cfgs = {"gem": {"provider": "gemini", "model": "gemini-1.5-flash", "fallback": "gemini-2.0-flash", "key_env": None},
            "good": {"provider": "x", "model": "m-ok", "key_env": None}}
    monkeypatch.setattr(llm, "load_model_configs", lambda: cfgs)

    def fake_call(self, prompt, **p):
        if self.model == "gemini-1.5-flash":
            raise RuntimeError("404 model not found")
        return "OK"
    monkeypatch.setattr(llm.LLMClient, "_call", fake_call)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    out = llm.check_models(["gem", "good"], tmp_path)
    assert out["gem"].startswith("fallback gemini-2.0-flash")
    assert out["good"] == "ok (m-ok)"
