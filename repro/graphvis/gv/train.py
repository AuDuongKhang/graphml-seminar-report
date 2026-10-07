"""Huấn luyện theo giai đoạn (curriculum / joint / fine-grained / text-only) bằng HF Trainer.

Mỗi giai đoạn là một W&B run trong cùng group = exp_id. Giai đoạn sau khởi tạo từ adapter của giai đoạn
trước (tiếp tục huấn luyện cùng LoRA + projector). Chạy lại cùng lệnh sẽ bỏ qua giai đoạn đã xong và
resume giai đoạn đang dở từ checkpoint gần nhất.
"""
import gc
import random
from pathlib import Path

from .data.build import read_rows
from .data.tasks import GRAPH_TASKS, OCR_TASKS

# nguồn dữ liệu -> (file jsonl, bộ lọc task, có ảnh)
SOURCES = {
    "dg": ("dg.jsonl", None, True),
    "dg_ocr": ("dg.jsonl", set(OCR_TASKS), True),
    "dg_graph": ("dg.jsonl", set(GRAPH_TASKS), True),
    "dv": ("dv.jsonl", None, True),
    "text": ("text.jsonl", None, False),
}


def plan_stages(cfg: dict) -> list:
    plan = []
    for st in cfg["stages"]:
        for s in st["data"]:
            if s not in SOURCES:
                raise ValueError(f"nguồn dữ liệu không hợp lệ: {s}")
        if len({SOURCES[s][2] for s in st["data"]}) > 1:
            raise ValueError(f"giai đoạn {st['name']} trộn dữ liệu có ảnh và không ảnh")
        plan.append(list(st["data"]))
    return plan


def load_stage_rows(data_dir: Path, sources: list, shuffle: bool = False, seed: int = 42,
                    limit: int | None = None) -> list:
    rows = []
    for s in sources:
        fname, tasks, _ = SOURCES[s]
        rows += [r for r in read_rows(Path(data_dir) / fname) if tasks is None or r["task"] in tasks]
    if shuffle:
        random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


def grad_accum(global_batch: int, per_device: int) -> int:
    if global_batch % per_device:
        raise ValueError("global_batch phải chia hết cho per_device_batch")
    return global_batch // per_device


class _Rows:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def experiment_id(cfg: dict, dry_run: bool) -> str:
    """Dry-run ghi vào thư mục riêng, không để lại final/ giả làm run thật bỏ qua giai đoạn."""
    return cfg["exp_id"] + ("-dryrun" if dry_run else "")


def single_gpu_guard(args, quant: str) -> None:
    """Trainer bọc mô hình bằng nn.DataParallel khi thấy >1 GPU (Kaggle 2xT4) — hỏng với bnb 4-bit + PEFT
    và làm batch thực tế nhân đôi. Ép 1 GPU cho QLoRA."""
    if quant == "4bit" and getattr(args, "_n_gpu", 1) > 1:
        args._n_gpu = 1


def prune_checkpoints(stage_dir: Path) -> None:
    """Sau khi đã lưu final/, xóa checkpoint-* (mỗi cái ~4 GB do trạng thái Adam) để không tràn đĩa Kaggle/Drive."""
    import shutil
    for ck in Path(stage_dir).glob("checkpoint-*"):
        shutil.rmtree(ck, ignore_errors=True)


def half_state_dict(sd: dict) -> dict:
    import torch
    return {k: (v.half() if torch.is_floating_point(v) else v) for k, v in sd.items()}


def _upload(folder: Path, hub_repo: str, path_in_repo: str):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(hub_repo, private=True, exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=hub_repo, path_in_repo=path_in_repo)


def run_stages(cfg: dict, dry_run: bool = False) -> Path:
    import torch
    from transformers import Trainer, TrainingArguments

    from repro.common.paths import out_root
    from repro.common.tracking import init_wandb

    from .model.collate import Collator
    from .model.load import ensure_adapter, load_model_and_processor, param_groups
    from .pipeline import split_dir

    plan_stages(cfg)
    exp = experiment_id(cfg, dry_run)
    out = out_root() / "graphvis" / exp
    data_dir = split_dir(cfg["dataset"], "train", cfg.get("data_tag", ""))
    prev = None
    for st in cfg["stages"]:
        sdir = out / st["name"]
        final = sdir / "final"
        if not dry_run and cfg.get("hub_repo"):
            ensure_adapter(str(final), cfg["hub_repo"])      # phiên mới: lấy lại giai đoạn đã xong từ HF Hub
        if (final / "adapter_config.json").exists():
            prev = final
            continue
        rows = load_stage_rows(data_dir, st["data"], shuffle=st.get("shuffle", False), seed=cfg.get("seed", 42),
                               limit=16 if dry_run else None)
        run = init_wandb(f"{exp}/{st['name']}", {**cfg, "stage": st["name"], "n_rows": len(rows), "dry_run": dry_run},
                         sdir, group=exp)
        model, proc = load_model_and_processor(
            image_mode=cfg.get("image_mode", "pad"), quant=cfg.get("quant", "4bit"), dtype=cfg.get("dtype", "fp16"),
            lora=None if prev else cfg.get("lora", {"r": 128, "alpha": 256, "dropout": 0.05}),
            adapter_path=str(prev) if prev else None, trainable_adapter=prev is not None)
        per_dev = cfg.get("per_device_batch", 1)
        args = TrainingArguments(
            output_dir=str(sdir), per_device_train_batch_size=per_dev,
            gradient_accumulation_steps=grad_accum(cfg.get("global_batch", 4), per_dev),
            num_train_epochs=cfg.get("epochs_per_stage", 1), max_steps=2 if dry_run else -1,
            learning_rate=cfg.get("lr", 2e-5), lr_scheduler_type="cosine", warmup_ratio=0.03, weight_decay=0.0,
            fp16=cfg.get("dtype", "fp16") == "fp16", bf16=cfg.get("dtype") == "bf16",
            logging_steps=10, save_steps=cfg.get("save_steps", 500), save_total_limit=2, report_to=["wandb"],
            run_name=f"{exp}/{st['name']}", remove_unused_columns=False, label_names=["labels"],
            dataloader_num_workers=2, seed=cfg.get("seed", 42), gradient_checkpointing=False)
        single_gpu_guard(args, cfg.get("quant", "4bit"))
        opt = torch.optim.AdamW(param_groups(model.named_parameters(), cfg.get("lr", 2e-5),
                                             cfg.get("projector_lr", 2e-5)))
        trainer = Trainer(model=model, args=args, train_dataset=_Rows(rows),
                          data_collator=Collator(proc, cfg.get("max_len", 2048), cfg.get("image_mode", "pad")),
                          optimizers=(opt, None))
        resume = any(sdir.glob("checkpoint-*"))
        trainer.train(resume_from_checkpoint=True if resume else None)
        from peft import get_peft_model_state_dict
        model.save_pretrained(str(final), state_dict=half_state_dict(get_peft_model_state_dict(model)))   # fp16: ~0,7 GB
        prune_checkpoints(sdir)
        if cfg.get("hub_repo") and not dry_run:
            _upload(final, cfg["hub_repo"], f"{exp}/{st['name']}")
        run.finish()
        prev = final
        del trainer, model, opt
        gc.collect()
        torch.cuda.empty_cache()
    return prev
