"""Nạp LLaVA-1.6-Mistral-7B (bản HF) theo cấu hình GraphVis: CLIP đóng băng, projector học toàn bộ,
LLM đóng băng + LoRA r=128/alpha=256 trên mọi lớp tuyến tính. Hỗ trợ QLoRA 4-bit cho GPU miễn phí (T4/P100).
"""
MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
PAD_PINPOINTS = [[336, 336]]
# Mọi nn.Linear của khối ngôn ngữ (đường dẫn module khác nhau giữa các bản transformers nên dùng regex)
LORA_TARGET_REGEX = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
PROJECTOR_KEY = "multi_modal_projector"


def param_groups(named_params, lr: float, projector_lr: float) -> list:
    lora, proj = [], []
    for name, p in named_params:
        if not p.requires_grad:
            continue
        (proj if PROJECTOR_KEY in name else lora).append(p)
    groups = [{"params": lora, "lr": lr, "weight_decay": 0.0}]
    if proj:
        groups.append({"params": proj, "lr": projector_lr, "weight_decay": 0.0})
    return groups


def torch_dtype(name: str):
    import torch
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def load_model_and_processor(model_id: str = MODEL_ID, image_mode: str = "pad", quant: str = "4bit",
                             dtype: str = "fp16", lora: dict | None = None, adapter_path: str | None = None,
                             trainable_adapter: bool = False, gradient_checkpointing: bool = True):
    """lora: {"r":128,"alpha":256,"dropout":0.05} để tạo adapter mới; adapter_path: nạp adapter đã huấn luyện
    (trainable_adapter=True khi giai đoạn sau tiếp tục huấn luyện từ adapter của giai đoạn trước)."""
    import torch
    from transformers import BitsAndBytesConfig, LlavaNextForConditionalGeneration, LlavaNextProcessor

    dt = torch_dtype(dtype)
    processor = LlavaNextProcessor.from_pretrained(model_id)
    qcfg = None
    if quant == "4bit":
        qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                  bnb_4bit_compute_dtype=dt,
                                  llm_int8_skip_modules=["vision_tower", PROJECTOR_KEY, "lm_head"])
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dt, quantization_config=qcfg, low_cpu_mem_usage=True,
        device_map={"": 0} if torch.cuda.is_available() else None)
    if image_mode == "pad":
        processor.image_processor.image_grid_pinpoints = PAD_PINPOINTS
        model.config.image_grid_pinpoints = PAD_PINPOINTS
    for n, p in model.named_parameters():
        p.requires_grad = False
    training = lora is not None or trainable_adapter
    if training and gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    if quant == "4bit" and training:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=trainable_adapter)
    elif lora is not None:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=lora.get("r", 128), lora_alpha=lora.get("alpha", 256),
                         lora_dropout=lora.get("dropout", 0.05), target_modules=LORA_TARGET_REGEX,
                         modules_to_save=[PROJECTOR_KEY], task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
    if training:
        # GradScaler (fp16) không unscale được gradient fp16 -> tham số học giữ fp32, forward chạy trong autocast
        for p in model.parameters():
            if p.requires_grad and p.dtype != torch.float32:
                p.data = p.data.float()
    return model, processor
