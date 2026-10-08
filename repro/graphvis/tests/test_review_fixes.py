"""Test tái hiện các lỗi do review toàn nhánh phát hiện (C1, I1, I2, I3, I8, I10)."""
from pathlib import Path
from types import SimpleNamespace

import torch

from gv.model.collate import Collator


# ---------------- C1: Kaggle 2xT4 -> Trainer bọc QLoRA bằng DataParallel ----------------
def test_single_gpu_guard_disables_dataparallel_for_4bit():
    from gv.train import single_gpu_guard
    args = SimpleNamespace(_n_gpu=2)
    single_gpu_guard(args, quant="4bit")
    assert args._n_gpu == 1
    args = SimpleNamespace(_n_gpu=2)
    single_gpu_guard(args, quant="none")
    assert args._n_gpu == 2


# ---------------- I1: train không có BOS nhưng eval có BOS ----------------
class FakeProc:
    def __init__(self):
        self.calls = []
        self.tokenizer = SimpleNamespace(padding_side="left", pad_token_id=0)

    def apply_chat_template(self, msgs, add_generation_prompt=False):
        text = "[INST] " + msgs[0]["content"][-1]["text"] + " [/INST]"
        if len(msgs) > 1:
            text += " " + msgs[1]["content"][0]["text"] + "</s> "
        return text

    def __call__(self, text, return_tensors="pt", padding=False, add_special_tokens=True, images=None):
        self.calls.append({"text": text, "add_special_tokens": add_special_tokens})
        texts = text if isinstance(text, list) else [text]
        ids = [[1] * add_special_tokens + [5] * len(t.split()) for t in texts]
        width = max(len(x) for x in ids)
        return {"input_ids": torch.tensor([x + [0] * (width - len(x)) for x in ids]),
                "attention_mask": torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x in ids])}


def test_collator_keeps_bos_like_inference_and_strips_trailing_space():
    proc = FakeProc()
    Collator(proc)([{"image": None, "prompt": "Q?", "answer": "A"}])
    assert all(c["add_special_tokens"] is not False for c in proc.calls)
    full = [c for c in proc.calls if isinstance(c["text"], list)][0]["text"][0]
    assert full.endswith("</s>")


# ---------------- I2: checkpoint chiếm hết đĩa ----------------
def test_prune_checkpoints_after_final(tmp_path):
    from gv.train import prune_checkpoints
    for name in ["checkpoint-500", "checkpoint-1000", "final"]:
        (tmp_path / name).mkdir()
    prune_checkpoints(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["final"]


def test_half_precision_state_dict():
    from gv.train import half_state_dict
    sd = half_state_dict({"a": torch.zeros(2, dtype=torch.float32), "b": torch.zeros(2, dtype=torch.int64)})
    assert sd["a"].dtype == torch.float16 and sd["b"].dtype == torch.int64


# ---------------- I3: adapter trên HF Hub không được tải về ở phiên mới ----------------
def test_ensure_adapter_downloads_missing_final(tmp_path, monkeypatch):
    from gv.model import load
    final = tmp_path / "out" / "graphvis" / "G1-csqa-curriculum" / "stage2_qa" / "final"
    seen = {}

    def fake_download(repo_id, allow_patterns, local_dir):
        seen.update(repo=repo_id, pattern=allow_patterns)
        d = Path(local_dir) / "G1-csqa-curriculum" / "stage2_qa"
        d.mkdir(parents=True)
        (d / "adapter_config.json").write_text("{}")
        return local_dir

    monkeypatch.setenv("GRAPHML_HUB_REPO", "me/adapters")
    monkeypatch.setattr(load, "_snapshot_download", fake_download)
    out = load.ensure_adapter(str(final))
    assert seen == {"repo": "me/adapters", "pattern": "G1-csqa-curriculum/stage2_qa/*"}
    assert (Path(out) / "adapter_config.json").exists()


def test_ensure_adapter_none_and_existing(tmp_path):
    from gv.model.load import ensure_adapter
    assert ensure_adapter(None) is None and ensure_adapter("none") is None
    (tmp_path / "adapter_config.json").write_text("{}")
    assert ensure_adapter(str(tmp_path)) == str(tmp_path)


# ---------------- I8: đánh giá đọc hình quá chậm ----------------
def test_sort_by_task_groups_long_outputs():
    from gv.eval.graph_reading import sort_by_task
    rows = [{"task": "triple_listing"}, {"task": "node_count"}, {"task": "triple_listing"}, {"task": "edge_count"}]
    assert [r["task"] for r in sort_by_task(rows)] == ["edge_count", "node_count", "triple_listing", "triple_listing"]


# ---------------- I10: dry-run ghi đè thư mục của run thật ----------------
def test_dry_run_uses_separate_experiment_id():
    from gv.train import experiment_id
    assert experiment_id({"exp_id": "G1-csqa-curriculum"}, dry_run=True) == "G1-csqa-curriculum-dryrun"
    assert experiment_id({"exp_id": "G1-csqa-curriculum"}, dry_run=False) == "G1-csqa-curriculum"


# ---------------- Rà soát lại: _setup_devices (cached) chạy sau guard sẽ đặt lại _n_gpu = 2 ----------------
def test_single_gpu_guard_survives_lazy_device_setup():
    from gv.train import single_gpu_guard

    class Args:
        _n_gpu = 0

        @property
        def device(self):          # mô phỏng TrainingArguments._setup_devices: đặt _n_gpu khi lần đầu truy cập
            if not getattr(self, "_setup", False):
                self._setup = True
                self._n_gpu = 2
            return "cuda:0"
    a = Args()
    single_gpu_guard(a, quant="4bit")
    assert a.device == "cuda:0" and a._n_gpu == 1


# ---------------- Rà soát lại: truyền state_dict đã lọc vào PeftModel.save_pretrained làm mất projector ----------------
def test_halve_adapter_file_keeps_every_tensor(tmp_path):
    from safetensors.torch import load_file, save_file

    from gv.train import halve_adapter_file
    f = tmp_path / "adapter_model.safetensors"
    save_file({"base_model.model.multi_modal_projector.linear_1.weight": torch.ones(2, 2),
               "base_model.model.language_model.layers.0.q_proj.lora_A.weight": torch.ones(2, 2),
               "step": torch.tensor([3])}, str(f))
    halve_adapter_file(tmp_path)
    sd = load_file(str(f))
    assert set(sd) == {"base_model.model.multi_modal_projector.linear_1.weight",
                       "base_model.model.language_model.layers.0.q_proj.lora_A.weight", "step"}
    assert sd["base_model.model.multi_modal_projector.linear_1.weight"].dtype == torch.float16
    assert sd["step"].dtype == torch.int64
