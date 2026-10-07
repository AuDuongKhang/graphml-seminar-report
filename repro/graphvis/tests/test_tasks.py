import json
import random

from gv.data.build import build_dataset
from gv.data.download import QAItem
from gv.data.retrieve import SubGraph
from gv.data.tasks import GRAPH_TASKS, OCR_TASKS, PROMPTS, TASKS, format_options, make_task_sample

SG = SubGraph(nodes=["choker", "jewelry", "box", "store"],
              triples=[("choker", "is a", "jewelry"), ("jewelry", "at location", "box"), ("box", "used for", "store")],
              qc={"choker"}, ac={"store"})
ITEM = QAItem(id="q1", question="Where would you put a choker?", choices=["store", "box", "neck"],
              labels=["A", "B", "C"], answer="B")


def test_prompt_pools_have_five_distinct_templates():
    for t in TASKS:
        assert len(set(PROMPTS[t])) == 5, t


def test_task_groups_partition_tasks():
    assert set(OCR_TASKS) | set(GRAPH_TASKS) == set(TASKS) and not set(OCR_TASKS) & set(GRAPH_TASKS)


def test_counts_and_degrees_are_consistent():
    _, a = make_task_sample("edge_count", SG, random.Random(0))
    assert a == "There are 3 edges in the graph."
    _, a = make_task_sample("node_count", SG, random.Random(0))
    assert a == "There are 4 nodes in the graph."
    _, a = make_task_sample("highest_degree", SG, random.Random(0))
    assert ('"jewelry"' in a or '"box"' in a) and a.endswith("with a degree of 2.")


def test_node_degree_prompt_substitutes_name():
    for seed in range(30):
        q, a = make_task_sample("node_degree", SG, random.Random(seed))
        assert "{node" not in q
        name = q.split('"')[1]
        expected = {"choker": 1, "jewelry": 2, "box": 2, "store": 1}[name]
        assert a == f'The degree of the node "{name}" is {expected}.'


def test_triple_listing_contains_all_triples():
    _, a = make_task_sample("triple_listing", SG, random.Random(0))
    for h, r, t in SG.triples:
        assert f"({h}, {r}, {t})" in a


def test_node_description_lists_all_nodes():
    _, a = make_task_sample("node_description", SG, random.Random(0))
    assert all(n in a for n in SG.nodes)


def test_format_options():
    assert format_options(ITEM) == "A. store\nB. box\nC. neck"


def _fake_render(sg, out_png, size_px=None):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_png.write_bytes(b"png")
    return out_png


def _read(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]


def test_build_dg_six_tasks_per_image(tmp_path):
    n = build_dataset("dg", [ITEM], {"q1": SG}, tmp_path / "img", tmp_path / "dg.jsonl", render_fn=_fake_render)
    rows = _read(tmp_path / "dg.jsonl")
    assert n == 6 and sorted(r["task"] for r in rows) == sorted(TASKS)
    assert all(r["image"] == "img/q1.png" and r["n_nodes"] == 4 and r["n_edges"] == 3 for r in rows)


def test_read_rows_resolves_relative_images(tmp_path):
    from gv.data.build import read_rows

    build_dataset("dv", [ITEM], {"q1": SG}, tmp_path / "img", tmp_path / "dv.jsonl", render_fn=_fake_render)
    rows = read_rows(tmp_path / "dv.jsonl")
    assert rows[0]["image"] == str(tmp_path / "img" / "q1.png")
    assert read_rows(tmp_path / "dv.jsonl", limit=0) == []


def test_build_dv_and_text(tmp_path):
    build_dataset("dv", [ITEM], {"q1": SG}, tmp_path / "img", tmp_path / "dv.jsonl", render_fn=_fake_render)
    build_dataset("text", [ITEM], {"q1": SG}, tmp_path / "img", tmp_path / "text.jsonl", render_fn=_fake_render)
    dv, tx = _read(tmp_path / "dv.jsonl")[0], _read(tmp_path / "text.jsonl")[0]
    assert dv["answer"] == "B" and dv["image"] and dv["prompt"].startswith("The image represents a knowledge graph")
    assert "A. store\nB. box\nC. neck" in dv["prompt"]
    assert tx["image"] is None and "image" not in tx["prompt"].lower() and tx["answer"] == "B"
    assert dv["question"] == tx["question"] == ITEM.question


def test_build_respects_max_items(tmp_path):
    items = [QAItem(id=f"q{i}", question="?", choices=["x", "y"], labels=["A", "B"], answer="A") for i in range(5)]
    graphs = {it.id: SG for it in items}
    n = build_dataset("dv", items, graphs, tmp_path / "img", tmp_path / "dv.jsonl", max_items=2, render_fn=_fake_render)
    assert n == 2
