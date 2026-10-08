from smkg.baselines import extract_gptner_spans, majority_vote, parse_json_list, top_k_similar, uniner_prompt
from smkg.llm import LLMClient


def test_extract_gptner_spans():
    assert extract_gptner_spans("细胞膜由@@磷脂##和@@蛋白质##构成，@@细胞膜##") == ["磷脂", "蛋白质", "细胞膜"]
    assert extract_gptner_spans("没有标记") == []


def test_parse_json_list():
    assert parse_json_list('["磷脂", "蛋白质"]') == ["磷脂", "蛋白质"]
    assert parse_json_list('Answer: ["a"] trailing') == ["a"]
    assert parse_json_list("not json") == []


def test_majority_vote_keeps_items_in_more_than_half():
    assert majority_vote([["a", "b"], ["a"], ["a", "c"], ["b"], ["a"]]) == ["a"]
    assert majority_vote([]) == []


def test_top_k_similar_prefers_shared_characters():
    pool = ["细胞膜的结构", "牛顿第二定律", "细胞核"]
    assert top_k_similar("细胞膜的功能", pool, k=2)[0] == "细胞膜的结构"
    assert len(top_k_similar("x", pool, k=5)) == 3


def test_uniner_prompt_format():
    p = uniner_prompt("文本", "Biology concept")
    assert p.startswith("A virtual assistant answers questions") and "Text: 文本" in p
    assert p.rstrip().endswith("ASSISTANT:") and "What describes Biology concept in the text?" in p


def test_completion_mode_uses_completions_endpoint(tmp_path):
    c = LLMClient("u", {"provider": "vllm", "model": "u", "mode": "completion"}, tmp_path, sleep=lambda s: None)
    seen = {}

    class Fake:
        class completions:
            @staticmethod
            def create(model, prompt, **p):
                seen["prompt"] = prompt
                return type("R", (), {"choices": [type("C", (), {"text": '["a"]'})()]})()
    c._client = Fake()
    assert c.chat("P").text == '["a"]' and seen["prompt"] == "P"
