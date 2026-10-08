import json
from pathlib import Path

import pytest
import yaml

from gv.train import SOURCES, grad_accum, load_stage_rows, plan_stages

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _write(p, rows):
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


@pytest.fixture
def data_dir(tmp_path):
    _write(tmp_path / "dg.jsonl", [{"id": f"q-{t}", "task": t, "image": "x.png", "prompt": "p", "answer": "a"}
                                   for t in ["node_description", "node_degree", "highest_degree",
                                             "node_count", "edge_count", "triple_listing"]])
    _write(tmp_path / "dv.jsonl", [{"id": "q", "task": "qa", "image": "x.png", "prompt": "p", "answer": "A"}])
    _write(tmp_path / "text.jsonl", [{"id": "q", "task": "qa", "image": None, "prompt": "p", "answer": "A"}])
    return tmp_path


def test_plan_stages_variants():
    assert plan_stages({"stages": [{"name": "s1", "data": ["dg"]}, {"name": "s2", "data": ["dv"]}]}) == [["dg"], ["dv"]]
    assert plan_stages({"stages": [{"name": "joint", "data": ["dg", "dv"]}]}) == [["dg", "dv"]]
    with pytest.raises(ValueError):
        plan_stages({"stages": [{"name": "x", "data": ["nope"]}]})
    with pytest.raises(ValueError):
        plan_stages({"stages": [{"name": "x", "data": ["dv", "text"]}]})   # trộn ảnh và không ảnh


def test_load_stage_rows_filters_task_groups(data_dir):
    ocr = load_stage_rows(data_dir, ["dg_ocr"])
    graph = load_stage_rows(data_dir, ["dg_graph"])
    assert {r["task"] for r in ocr} == {"node_description", "triple_listing"}
    assert len(graph) == 4
    joint = load_stage_rows(data_dir, ["dg", "dv"], shuffle=True, seed=1)
    assert len(joint) == 7 and joint != load_stage_rows(data_dir, ["dg", "dv"])


def test_load_stage_rows_limit(data_dir):
    assert len(load_stage_rows(data_dir, ["dg"], limit=2)) == 2


def test_grad_accum():
    assert grad_accum(global_batch=4, per_device=1) == 4
    assert grad_accum(global_batch=4, per_device=2) == 2
    with pytest.raises(ValueError):
        grad_accum(global_batch=4, per_device=3)


def test_all_configs_are_valid():
    files = sorted(CONFIG_DIR.glob("*.yaml"))
    assert len(files) >= 7
    for f in files:
        cfg = yaml.safe_load(f.read_text(encoding="utf-8"))
        assert cfg["exp_id"] and cfg["dataset"] in ("csqa", "obqa")
        plan_stages(cfg)
        assert set(SOURCES) >= {s for st in cfg["stages"] for s in st["data"]}
