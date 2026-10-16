"""G0 — dựng dữ liệu: truy hồi đồ thị con, vẽ ảnh, sinh dg/dv/text cho một (dataset, split)."""
import json
from pathlib import Path

from repro.common.paths import data_root

from .data import download as dl
from .data import hub
from .data.build import build_dataset
from .data.retrieve import SubGraph, build_question_graph, load_cpnet_csv

# Số ảnh dùng cho D_g và số câu cho D_v ở split huấn luyện; split đánh giá dùng đủ (trừ tiny)
SCALES = {
    "tiny": {"train": {"dg": 40, "dv": 200}, "eval": {"dg": 20, "dv": 100}},
    "lite": {"train": {"dg": 600, "dv": 2000}, "eval": {"dg": None, "dv": None}},
    "full": {"train": {"dg": None, "dv": None}, "eval": {"dg": None, "dv": None}},
}
TRAIN_SPLITS = {"train"}


def scale_limits(scale: str, role: str) -> dict:
    return dict(SCALES[scale][role])


def save_graphs(graphs: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid, sg in graphs.items():
            f.write(json.dumps({"id": qid, "nodes": sg.nodes, "triples": sg.triples,
                                "qc": sorted(sg.qc), "ac": sorted(sg.ac)}, ensure_ascii=False) + "\n")


def load_graphs(path: Path) -> dict:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out[d["id"]] = SubGraph(nodes=d["nodes"], triples=[tuple(t) for t in d["triples"]],
                                    qc=set(d["qc"]), ac=set(d["ac"]))
    return out


def split_dir(dataset: str, split: str, tag: str = "") -> Path:
    return data_root() / "graphvis" / dataset / (split + (f"_{tag}" if tag else ""))


def build_data(dataset: str, split: str, scale: str = "lite", image_size: int | None = None,
               tag: str = "", max_nodes: int = 80, use_wandb: bool = True, hub_repo: str | None = None) -> Path:
    out = split_dir(dataset, split, tag)
    remote = hub.archive_path(scale, dataset, out.name) if hub_repo else None
    if remote:
        hub.ensure_repo(hub_repo)                    # sai HUB_REPO/quyền: dừng ngay, trước khi dựng ~10 phút
    if remote and not (out / hub.DONE).exists() and hub.pull(hub_repo, remote, out):
        return out                                   # bản đã dựng ở lượt trước, không truy hồi/vẽ lại
    paths = dl.download_qagnn(data_root() / "qagnn")
    out.mkdir(parents=True, exist_ok=True)
    gpath = out / "graphs.jsonl"
    items = dl.load_qa(paths, dataset, split)
    if gpath.exists():
        graphs = load_graphs(gpath)
    else:
        cp = load_cpnet_csv(paths["cpnet_csv"], paths["concept_txt"], cache=data_root() / "qagnn" / "cpnet.npz")
        groups = dl.group_grounded(items, dl.load_grounded(paths, dataset, split))
        graphs = {it.id: build_question_graph(cp, g, max_nodes=max_nodes) for it, g in zip(items, groups)}
        save_graphs(graphs, gpath)
    lim = scale_limits(scale, "train" if split in TRAIN_SPLITS else "eval")
    img = out / "images"
    stats = {
        "dg": build_dataset("dg", items, graphs, img, out / "dg.jsonl", max_items=lim["dg"], size_px=image_size),
        "dv": build_dataset("dv", items, graphs, img, out / "dv.jsonl", max_items=lim["dv"], size_px=image_size),
        "text": build_dataset("text", items, graphs, img, out / "text.jsonl", max_items=lim["dv"]),
        "n_questions": len(items),
        "empty_graphs": sum(1 for g in graphs.values() if not g.nodes),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    if use_wandb:
        _log_data_run(dataset, split, tag, scale, image_size, graphs, img, stats, out)
    if remote:
        hub.push(hub_repo, remote, out)
    return out


def _log_data_run(dataset, split, tag, scale, image_size, graphs, img_dir, stats, out):
    import wandb

    from repro.common.tracking import init_wandb

    exp = f"G0-data-{dataset}-{split}" + (f"-{tag}" if tag else "")
    run = init_wandb(exp, {"dataset": dataset, "split": split, "scale": scale, "image_size": image_size},
                     out / "wandb_run", group="G0")
    nodes = [len(g.nodes) for g in graphs.values()]
    edges = [len(g.triples) for g in graphs.values()]
    run.log({"n_nodes": wandb.Histogram(nodes), "n_edges": wandb.Histogram(edges),
             "n_nodes_table": wandb.Table(data=[[n, e] for n, e in zip(nodes, edges)], columns=["n_nodes", "n_edges"])})
    run.summary.update(stats)
    samples = sorted(Path(img_dir).glob("*.png"))[:20]
    run.log({"examples": [wandb.Image(str(p), caption=p.stem) for p in samples]})
    run.finish()
