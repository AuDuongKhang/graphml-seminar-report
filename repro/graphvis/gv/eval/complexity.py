"""Ảnh hưởng độ phức tạp ảnh đồ thị: C1 (độ chính xác theo số nút) và C2 (theo độ phân giải ảnh)."""
import json

from .mcqa import BINS, node_bin


def combine_bins(preds_by_exp: dict) -> list:
    """{exp: [{"n_nodes","gold","pred"}...]} -> các dòng [exp, bin, accuracy, count]."""
    table = []
    for exp, rows in preds_by_exp.items():
        tot, ok = {}, {}
        for r in rows:
            b = node_bin(r["n_nodes"])
            tot[b] = tot.get(b, 0) + 1
            ok[b] = ok.get(b, 0) + int(r["pred"] == r["gold"])
        for _, _, b in BINS:
            if b in tot:
                table.append([exp, b, ok[b] / tot[b], tot[b]])
    return table


def load_predictions(exp_id: str, base, api=None) -> list:
    """predictions.jsonl của một lần eval-mcqa; thiếu file (lượt Kaggle sau, /kaggle/working đã trống) thì đọc
    bảng `predictions` của run W&B cùng tên, lấy run hoàn tất gần nhất."""
    import os
    from pathlib import Path

    f = Path(base) / exp_id / "predictions.jsonl"
    if f.exists():
        return [json.loads(l) for l in open(f, encoding="utf-8")]
    if api is None:
        import wandb
        api = wandb.Api()
    from repro.common.tracking import DEFAULT_PROJECT
    path = f"{os.environ.get('WANDB_ENTITY') or api.default_entity}/{os.environ.get('WANDB_PROJECT', DEFAULT_PROJECT)}"
    runs = [r for r in api.runs(path, filters={"display_name": exp_id}) if r.state == "finished"]
    if not runs:
        raise FileNotFoundError(f"{exp_id}: không có {f} và không có run W&B hoàn tất trong {path}")
    run = max(runs, key=lambda r: r.created_at)
    art = next(a for a in run.logged_artifacts() if a.type == "run_table" and "-predictions" in a.name)
    table = art.get("predictions")
    rows = [dict(zip(table.columns, d)) for d in table.data]
    return [{"id": r["id"], "n_nodes": r["n_nodes"], "gold": r["gold"], "pred": r["pred_letter"]} for r in rows]


def c1_report(eval_exp_ids: list, exp_id: str = "C1-accuracy-by-nodes") -> list:
    """Gộp predictions.jsonl của các lần eval-mcqa (vd base-image, GraphVis, C3 text-only) thành một bảng W&B."""
    import wandb

    from repro.common.paths import out_root
    from repro.common.tracking import init_wandb

    base = out_root() / "graphvis" / "eval"
    api = None if all((base / e / "predictions.jsonl").exists() for e in eval_exp_ids) else wandb.Api()
    preds = {e: load_predictions(e, base, api=api) for e in eval_exp_ids}
    table = combine_bins(preds)
    run = init_wandb(exp_id, {"sources": eval_exp_ids}, base / exp_id, group="C1")
    t = wandb.Table(columns=["model", "n_nodes_bin", "accuracy", "count"], data=table)
    run.log({"accuracy_by_nodes": t,
             "accuracy_by_nodes_chart": wandb.plot.line(t, "n_nodes_bin", "accuracy", stroke="model",
                                                        title="CSQA accuracy theo số nút đồ thị con")})
    run.finish()
    return table


def resolution_sweep(adapter: str | None, exp_prefix: str, sizes=(50, 112, 224, 336, 672),
                     image_modes=("pad", "anyres"), dataset: str = "csqa", split: str = "dev",
                     max_eval: int | None = None, batch_size: int = 8, quant: str = "4bit", dtype: str = "fp16") -> dict:
    """C2: giữ nguyên mô hình, thu nhỏ ảnh đồ thị về size x size trước khi đưa vào processor."""
    from ..model.load import load_model_and_processor
    from .mcqa import eval_mcqa

    results = {}
    for mode in image_modes:
        model, proc = load_model_and_processor(image_mode=mode, quant=quant, dtype=dtype,
                                               adapter_path=adapter if adapter not in (None, "none") else None)
        for s in list(sizes) + [None]:
            name = f"{exp_prefix}-{mode}-{s or 'orig'}"
            results[name] = eval_mcqa(dataset, split, "image", adapter, name, group="C2", max_eval=max_eval,
                                      batch_size=batch_size, quant=quant, dtype=dtype, image_mode=mode,
                                      image_size=s, model=model, proc=proc)["accuracy"]
        del model
    return results
