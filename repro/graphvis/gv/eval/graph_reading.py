"""Chấm 6 tác vụ đọc ảnh đồ thị (Table 4 của bài).

Số / tên: khớp đúng (lấy số nguyên đầu tiên trong câu trả lời). Liệt kê nút và liệt kê bộ ba: tỉ lệ
phần tử đúng xuất hiện trong câu trả lời, lấy trung bình theo mẫu ("average accuracy of each test example").
"""
import re

from ..data.tasks import degrees

_INT = re.compile(r"-?\d+")


def _first_int(text: str):
    m = _INT.search(text or "")
    return int(m.group()) if m else None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def score_graph_task(task: str, pred: str, sg, gold: str) -> float:
    p = _norm(pred)
    if task in ("node_count", "edge_count", "node_degree"):
        want = _first_int(gold.split(" is ")[-1] if task == "node_degree" else gold)
        return float(want is not None and _first_int(pred) == want)
    if task == "highest_degree":
        deg = degrees(sg)
        if not deg:
            return 0.0
        top = max(deg.values())
        names = [n for n, d in deg.items() if d == top]
        name_ok = any(f'"{n.lower()}"' in p or re.search(rf"\b{re.escape(n.lower())}\b", p) for n in names)
        ints = [int(x) for x in _INT.findall(p)]
        return float(name_ok and top in ints)
    if task == "node_description":
        if not sg.nodes:
            return 0.0
        return sum(1 for n in sg.nodes if n.lower() in p) / len(sg.nodes)
    if task == "triple_listing":
        if not sg.triples:
            return 0.0
        return sum(1 for h, r, t in sg.triples if _norm(f"({h}, {r}, {t})") in p) / len(sg.triples)
    raise ValueError(task)


def sort_by_task(rows: list) -> list:
    """Gom các tác vụ có câu trả lời dài (liệt kê) vào cùng batch để batch ngắn không phải chờ 384 token."""
    return sorted(rows, key=lambda r: r["task"])


def eval_graph_reading(adapter: str | None, exp_id: str, group: str | None = None, dataset: str = "csqa",
                       split: str = "dev", max_eval: int | None = None, batch_size: int = 4, quant: str = "4bit",
                       dtype: str = "fp16", image_mode: str = "pad") -> dict:
    """Tập kiểm tra = dg.jsonl dựng từ split đánh giá (1 câu mỗi tác vụ cho mỗi ảnh), như bài dùng ảnh từ CSQA test."""
    from collections import defaultdict

    import wandb

    from repro.common.paths import out_root
    from repro.common.tracking import init_wandb

    from ..data.build import read_rows
    from ..model.load import load_model_and_processor
    from ..pipeline import load_graphs, split_dir
    from .generate import generate
    from .mcqa import node_bin

    d = split_dir(dataset, split)
    rows = sort_by_task(read_rows(d / "dg.jsonl", limit=max_eval))
    graphs = load_graphs(d / "graphs.jsonl")
    out = out_root() / "graphvis" / "eval" / exp_id
    run = init_wandb(exp_id, {"adapter": adapter, "dataset": dataset, "split": split, "n": len(rows)}, out, group=group)
    model, proc = load_model_and_processor(image_mode=image_mode, quant=quant, dtype=dtype,
                                           adapter_path=adapter if adapter not in (None, "none") else None)
    preds = generate(model, proc, rows, image_mode=image_mode, batch_size=batch_size, max_new_tokens=384, dtype=dtype)
    per_task, per_bin = defaultdict(list), defaultdict(list)
    data = []
    for r, p in zip(rows, preds):
        s = score_graph_task(r["task"], p, graphs[r["id"].rsplit("-", 1)[0]], r["answer"])
        per_task[r["task"]].append(s)
        per_bin[(r["task"], node_bin(r["n_nodes"]))].append(s)
        data.append([r["id"], r["task"], r["n_nodes"], r["answer"], p, s])
    summary = {f"acc/{t}": 100 * sum(v) / len(v) for t, v in per_task.items()}
    summary.update({f"acc_by_nodes/{t}/{b}": 100 * sum(v) / len(v) for (t, b), v in per_bin.items()})
    run.log({"predictions": wandb.Table(columns=["id", "task", "n_nodes", "gold", "pred", "score"], data=data)})
    run.summary.update(summary)
    run.finish()
    return summary
