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


# ---------------- Bộ nhớ T4 (log lượt 2: hết VRAM ở bước 8/895 của G1b) ----------------
def test_qlora_uses_paged_8bit_adamw(monkeypatch):
    """Trạng thái AdamW fp32 của 336M tham số LoRA tốn ~2,7 GB; bản 8-bit phân trang còn ~0,7 GB."""
    import sys
    import types

    from gv.train import make_optimizer

    made = {}
    bnb = types.ModuleType("bitsandbytes")
    bnb.optim = types.SimpleNamespace(PagedAdamW8bit=lambda groups: made.setdefault("groups", groups))
    monkeypatch.setitem(sys.modules, "bitsandbytes", bnb)
    groups = [{"params": [], "lr": 1e-4, "weight_decay": 0.0}]
    make_optimizer(groups, "paged_adamw_8bit")
    assert made["groups"] is groups


def test_make_optimizer_plain_adamw_and_unknown():
    import torch

    from gv.train import default_optim, make_optimizer

    p = torch.nn.Parameter(torch.zeros(2))
    assert isinstance(make_optimizer([{"params": [p], "lr": 1e-4}], "adamw"), torch.optim.AdamW)
    with pytest.raises(ValueError):
        make_optimizer([], "sgd")
    assert default_optim({"quant": "4bit"}) == "paged_adamw_8bit"
    assert default_optim({"quant": "none"}) == "adamw"           # LoRA bf16 trên A100: AdamW như bài
    assert default_optim({"quant": "4bit", "optim": "adamw"}) == "adamw"


def test_failed_training_marks_wandb_run_failed():
    """Hết VRAM giữa chừng mà W&B vẫn ghi 'finished' -> dễ tưởng giai đoạn đã xong."""
    from gv.train import train_or_mark_failed

    class Trainer:
        def train(self, resume_from_checkpoint=None):
            raise RuntimeError("CUDA out of memory")

    class Run:
        exit_code = None

        def finish(self, exit_code=None):
            self.exit_code = exit_code

    run = Run()
    with pytest.raises(RuntimeError, match="out of memory"):
        train_or_mark_failed(Trainer(), run, resume=False)
    assert run.exit_code == 1


def test_cli_enables_expandable_cuda_segments():
    """Log lượt 2: ~730 MiB bị giữ nhưng không dùng được (phân mảnh) lúc hết VRAM."""
    import os

    import gv.cli  # noqa: F401
    assert "expandable_segments:True" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
