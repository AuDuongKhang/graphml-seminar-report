"""Ảnh hưởng độ phức tạp ảnh đồ thị: C1 (độ chính xác theo số nút) và C2 (theo độ phân giải ảnh)."""
import json

from .generate import _free_cuda
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


def wandb_path(api) -> str:
    import os

    from repro.common.tracking import DEFAULT_PROJECT
    return f"{os.environ.get('WANDB_ENTITY') or api.default_entity}/{os.environ.get('WANDB_PROJECT', DEFAULT_PROJECT)}"


def latest_finished_run(exp_id: str, api):
    """Run W&B hoàn tất gần nhất mang tên exp_id trong project hiện hành (WANDB_ENTITY/WANDB_PROJECT)."""
    runs = [r for r in api.runs(wandb_path(api), filters={"display_name": exp_id}) if r.state == "finished"]
    return max(runs, key=lambda r: r.created_at) if runs else None


def done_accuracy(exp_id: str, api):
    """Độ chính xác đã ghi của một điểm đánh giá, None nếu chưa có. Run hỏng giữa chừng có thể vẫn ghi
    'finished' nên chỉ tin run có accuracy trong summary."""
    if api is None:
        return None
    try:
        run = latest_finished_run(exp_id, api)
    except Exception:                                # không có mạng/đăng nhập: coi như chưa chạy
        return None
    if run is None:
        return None
    return getattr(run.summary, "_json_dict", run.summary).get("accuracy")


def load_predictions(exp_id: str, base, api=None) -> list:
    """predictions.jsonl của một lần eval-mcqa; thiếu file (lượt Kaggle sau, /kaggle/working đã trống) thì đọc
    bảng `predictions` của run W&B cùng tên, lấy run hoàn tất gần nhất."""
    from pathlib import Path

    f = Path(base) / exp_id / "predictions.jsonl"
    if f.exists():
        return [json.loads(l) for l in open(f, encoding="utf-8")]
    if api is None:
        import wandb
        api = wandb.Api()
    run = latest_finished_run(exp_id, api)
    if run is None:
        raise FileNotFoundError(f"{exp_id}: không có {f} và không có run W&B hoàn tất trong {wandb_path(api)}")
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
                     max_eval: int | None = None, batch_size: int = 8, quant: str = "4bit", dtype: str = "fp16",
                     api="auto") -> dict:
    """C2: giữ nguyên mô hình, thu nhỏ ảnh đồ thị về size x size trước khi đưa vào processor.
    Điểm đã có accuracy trên W&B (lượt trước) được dùng lại, chế độ ảnh không còn điểm nào thì không nạp mô hình."""
    from ..model import load
    from . import mcqa

    if api == "auto":
        try:
            import wandb
            api = wandb.Api()
        except Exception:
            api = None
    results = {}
    for mode in image_modes:
        todo = []
        for s in list(sizes) + [None]:
            name = f"{exp_prefix}-{mode}-{s or 'orig'}"
            acc = done_accuracy(name, api)
            if acc is None:
                todo.append((s, name))
            else:
                results[name] = acc
                print(f"đã có trên W&B, bỏ qua: {name} = {acc}")
        if not todo:
            continue
        model, proc = load.load_model_and_processor(image_mode=mode, quant=quant, dtype=dtype,
                                                    adapter_path=adapter if adapter not in (None, "none") else None)
        for s, name in todo:
            results[name] = mcqa.eval_mcqa(dataset, split, "image", adapter, name, group="C2", max_eval=max_eval,
                                           batch_size=batch_size, quant=quant, dtype=dtype, image_mode=mode,
                                           image_size=s, model=model, proc=proc)["accuracy"]
        del model, proc
        _free_cuda()                # không để mô hình của chế độ trước nằm trên GPU khi nạp chế độ sau
    return results
