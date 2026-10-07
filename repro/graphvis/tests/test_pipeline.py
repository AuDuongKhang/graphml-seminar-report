from gv.data.retrieve import SubGraph
from gv.pipeline import SCALES, load_graphs, save_graphs, scale_limits


def test_graphs_roundtrip(tmp_path):
    g = {"q1": SubGraph(nodes=["a", "b"], triples=[("a", "is a", "b")], qc={"a"}, ac={"b"})}
    save_graphs(g, tmp_path / "graphs.jsonl")
    back = load_graphs(tmp_path / "graphs.jsonl")
    assert back["q1"].nodes == ["a", "b"] and back["q1"].triples == [("a", "is a", "b")]
    assert back["q1"].qc == {"a"}


def test_scale_limits():
    assert set(SCALES) == {"tiny", "lite", "full"}
    assert scale_limits("full", "train") == {"dg": None, "dv": None}
    assert scale_limits("lite", "train") == {"dg": 600, "dv": 2000}
    assert scale_limits("lite", "eval") == {"dg": None, "dv": None}   # đánh giá trên đủ dev/test
    assert scale_limits("tiny", "eval")["dv"] == 100
