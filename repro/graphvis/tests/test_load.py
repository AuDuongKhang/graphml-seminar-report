import re

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
