"""Dựng các tập jsonl huấn luyện/đánh giá từ câu hỏi QA và đồ thị con.

Mỗi dòng: {"id", "image" (đường dẫn hoặc None), "prompt", "answer", "task", "n_nodes", "n_edges"}.
- dg:   6 mẫu đọc ảnh đồ thị cho mỗi ảnh (giai đoạn 1)
- dv:   câu hỏi QA gốc kèm ảnh đồ thị (giai đoạn 2)
- text: câu hỏi QA không có ảnh (đối chứng C3, mô hình gốc chế độ text)
"""
import json
import random
from pathlib import Path

from .render import render_graph
from .tasks import QA_PROMPT, TASKS, TEXT_PROMPT, format_options, make_task_sample


def build_dataset(kind: str, items: list, graphs: dict, img_dir: Path, out_jsonl: Path,
                  tasks=TASKS, seed: int = 42, max_items: int | None = None,
                  size_px: int | None = None, render_fn=render_graph) -> int:
    rng = random.Random(seed)
    chosen = list(items)
    if max_items is not None and max_items < len(chosen):
        chosen = rng.sample(chosen, max_items)
    img_dir, out_jsonl = Path(img_dir), Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for it in chosen:
            sg = graphs[it.id]
            meta = {"n_nodes": len(sg.nodes), "n_edges": len(sg.triples)}
            image = None
            if kind in ("dg", "dv"):
                png = img_dir / f"{it.id}.png"
                if not png.exists():
                    render_fn(sg, png, size_px=size_px)
                image = str(png)
            if kind == "dg":
                rows = []
                for task in tasks:
                    if task in ("node_degree", "highest_degree") and not sg.nodes:
                        continue
                    q, a = make_task_sample(task, sg, rng)
                    rows.append({"id": f"{it.id}-{task}", "image": image, "prompt": q, "answer": a, "task": task, **meta})
            elif kind == "dv":
                rows = [{"id": it.id, "image": image, "answer": it.answer, "task": "qa",
                         "prompt": QA_PROMPT.format(question=it.question, options=format_options(it)), **meta}]
            elif kind == "text":
                rows = [{"id": it.id, "image": None, "answer": it.answer, "task": "qa",
                         "prompt": TEXT_PROMPT.format(question=it.question, options=format_options(it)), **meta}]
            else:
                raise ValueError(f"unknown kind {kind}")
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
    return n
