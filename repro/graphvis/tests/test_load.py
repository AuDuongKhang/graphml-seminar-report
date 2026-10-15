import re
import sys
import types

import torch

from gv.model.load import LORA_TARGET_REGEX, param_groups


def test_lora_targets_only_language_model_linears():
    rx = re.compile(LORA_TARGET_REGEX)
    yes = ["model.language_model.layers.0.self_attn.q_proj", "language_model.model.layers.31.mlp.down_proj",
           "model.language_model.layers.3.mlp.gate_proj"]
    no = ["model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj",
          "model.multi_modal_projector.linear_1", "lm_head", "model.language_model.embed_tokens"]
    assert all(rx.fullmatch(n) for n in yes)
    assert not any(rx.fullmatch(n) for n in no)


def test_param_groups_give_projector_its_own_lr():
    lora = torch.nn.Parameter(torch.zeros(2))
    proj = torch.nn.Parameter(torch.zeros(2))
    frozen = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
    named = [("base_model.model.language_model.layers.0.q_proj.lora_A.default.weight", lora),
             ("base_model.model.multi_modal_projector.modules_to_save.default.linear_1.weight", proj),
             ("base_model.model.vision_tower.x", frozen)]
    groups = param_groups(named, lr=1e-4, projector_lr=2e-5)
    assert [g["lr"] for g in groups] == [1e-4, 2e-5]
    assert groups[0]["params"] == [lora] and groups[1]["params"] == [proj]
    assert all(g["weight_decay"] == 0.0 for g in groups)


class _FakeLlava(torch.nn.Module):
    """Thay LlavaNext: HF bỏ qua gradient checkpointing khi module ở eval() (mặc định sau from_pretrained)."""

    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(image_grid_pinpoints=None)
        self.lin = torch.nn.Linear(2, 2)
        self.checkpointing = False

    @classmethod
    def from_pretrained(cls, *a, **k):
        return cls().eval()

    def gradient_checkpointing_enable(self, **k):
        self.checkpointing = True

    def enable_input_require_grads(self):
        pass


def _patch_hf(monkeypatch):
    """transformers/peft giả trong sys.modules: test chạy cả trên máy không có LlavaNext/peft."""
    proc = types.SimpleNamespace(image_processor=types.SimpleNamespace())
    hf = types.ModuleType("transformers")
    hf.BitsAndBytesConfig = lambda **k: k
    hf.LlavaNextForConditionalGeneration = _FakeLlava
    hf.LlavaNextProcessor = types.SimpleNamespace(from_pretrained=lambda *a, **k: proc)
    pf = types.ModuleType("peft")
    pf.LoraConfig = lambda **k: k
    pf.get_peft_model = lambda m, cfg: m
    monkeypatch.setitem(sys.modules, "transformers", hf)
    monkeypatch.setitem(sys.modules, "peft", pf)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def test_training_model_is_returned_in_train_mode(monkeypatch):
    """Gọi forward ngay sau khi nạp (test GPU) phải dùng gradient checkpointing như trong Trainer, nếu không
    LLaVA 7B giữ toàn bộ activation và hết VRAM trên T4 (log Kaggle lượt 1)."""
    from gv.model.load import load_model_and_processor

    _patch_hf(monkeypatch)
    model, _ = load_model_and_processor(quant="none", dtype="fp32", lora={"r": 4, "alpha": 8})
    assert model.training and model.checkpointing


def test_inference_model_stays_in_eval_mode(monkeypatch):
    from gv.model.load import load_model_and_processor

    _patch_hf(monkeypatch)
    model, _ = load_model_and_processor(quant="none", dtype="fp32")
    assert not model.training and not model.checkpointing
