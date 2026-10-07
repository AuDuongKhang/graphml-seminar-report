from smkg.llm import LLMClient, load_model_configs, parse_concepts, strip_think
from smkg.prompts import EXTRACT, FEEDBACK, REFINE, fill


def test_parse_concepts_handles_garbage():
    assert parse_concepts("-细胞膜\n- 磷脂\n") == (["细胞膜", "磷脂"], "ok")
    assert parse_concepts("• 叶绿体") == (["叶绿体"], "ok")
    assert parse_concepts("Here are the concepts: none")[1] == "format_error"
    assert parse_concepts(None)[1] == "no_response"
    assert parse_concepts("")[1] == "no_response"
    assert parse_concepts("- \n-  ")[1] == "format_error"


def test_strip_think():
    assert strip_think("<think>hmm</think>\n- a") == "- a"
    assert strip_think("- a") == "- a"


def _client(tmp_path, fn):
    c = LLMClient("fake", {"provider": "fake", "model": "fake"}, tmp_path, sleep=lambda s: None)
    c._call = fn
    return c


def test_client_counts_timeout_without_raising(tmp_path):
    def boom(prompt, **p):
        raise TimeoutError()
    c = _client(tmp_path, boom)
    r = c.chat("x")
    assert r.text is None and r.error == "timeout" and c.counters["timeout"] == 1


def test_client_counts_empty_as_no_response(tmp_path):
    c = _client(tmp_path, lambda prompt, **p: "  ")
    r = c.chat("x")
    assert r.text is None and r.error == "no_response" and c.counters["no_response"] == 1


def test_client_retries_then_succeeds(tmp_path):
    calls = []

    def flaky(prompt, **p):
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("503")
        return "- a"
    c = _client(tmp_path, flaky)
    assert c.chat("x").text == "- a" and len(calls) == 3


def test_cache_hit_and_errors_not_cached(tmp_path):
    calls = []
    c = _client(tmp_path, lambda prompt, **p: calls.append(1) or "-a")
    c.chat("p")
    r = c.chat("p")
    assert r.cached and len(calls) == 1
    c2 = _client(tmp_path, lambda prompt, **p: calls.append(1) or "-a")   # cache bền qua phiên
    assert c2.chat("p").cached and len(calls) == 1


def test_model_configs_cover_paper_and_open_models():
    cfg = load_model_configs()
    for name in ["gpt-4o", "deepseek-v3", "gemini-1.5-flash", "qwen3-14b", "gemma3-27b", "phi3-medium",
                 "qwen3-4b", "gemma3-4b", "phi3-mini", "qwen3-8b", "gemma3-12b", "phi3-small"]:
        assert name in cfg, name


def test_prompts_fill_subject_and_lists():
    p = fill(EXTRACT, subject="Biology")
    assert "Biology" in p and "{subject}" not in p
    r = fill(REFINE, subject="Biology", feedback="f", candidates="['a']")
    assert "['a']" in r and "{" not in r.replace("{subject}", "")
    assert "{" not in fill(FEEDBACK, subject="Physics")
