"""CLI GraphVis. Chạy từ thư mục repro/graphvis: python -m gv.cli <lệnh> ...  (xem README.md)."""
import argparse
import os
import sys
from pathlib import Path

# phải đặt trước khi import torch: giảm phân mảnh VRAM khi độ dài chuỗi thay đổi giữa các mẫu (T4 15 GB)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# cho phép `import repro.common...` khi chạy từ repro/graphvis
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def _add_build_data(sub):
    p = sub.add_parser("build-data", help="G0: truy hồi đồ thị con, vẽ ảnh, sinh dg/dv/text")
    p.add_argument("--dataset", choices=["csqa", "obqa"], required=True)
    p.add_argument("--split", required=True, help="csqa: train|dev; obqa: train|test")
    p.add_argument("--scale", choices=["tiny", "lite", "full"], default="lite")
    p.add_argument("--image_size", type=int, default=None, help="vd 50 cho thí nghiệm ảnh độ phân giải thấp (G7)")
    p.add_argument("--tag", default="", help="hậu tố thư mục, vd lowres50")
    p.add_argument("--max_nodes", type=int, default=80)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--hub_repo", default=os.environ.get("GRAPHML_HUB_REPO") or None,
                   help="repo HF Hub lưu bản dữ liệu đã dựng (mặc định GRAPHML_HUB_REPO); có sẵn thì tải về thay vì dựng lại")

    def run(a):
        from gv.pipeline import build_data
        out = build_data(a.dataset, a.split, a.scale, a.image_size, a.tag, a.max_nodes, not a.no_wandb, a.hub_repo)
        print(f"data -> {out}")

    p.set_defaults(func=run)


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # help tiếng Việt trên console Windows
    ap = argparse.ArgumentParser(prog="gv")
    sub = ap.add_subparsers(dest="cmd", required=True)
    _add_build_data(sub)
    for register in _EXTRA:
        register(sub)
    a = ap.parse_args(argv)
    a.func(a)


def _add_train(sub):
    p = sub.add_parser("train", help="huấn luyện theo các giai đoạn trong file config (G1, G4, G7, C3)")
    p.add_argument("--config", required=True)
    p.add_argument("--dry_run", action="store_true", help="16 mẫu, 2 bước mỗi giai đoạn — kiểm tra nhanh")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="ghi đè config, vd quant=none dtype=bf16")

    def run(a):
        import yaml
        from gv.train import run_stages
        cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
        for kv in a.set:
            k, v = kv.split("=", 1)
            cfg[k] = yaml.safe_load(v)
        print(f"adapter -> {run_stages(cfg, dry_run=a.dry_run)}")

    p.set_defaults(func=run)


def _model_args(p):
    p.add_argument("--adapter", default="none", help="thư mục adapter (…/final) hoặc none = mô hình gốc")
    p.add_argument("--exp_id", required=True, help="tên W&B run, vd G1a-base-csqa-text")
    p.add_argument("--group", default=None, help="W&B group, vd G1")
    p.add_argument("--max_eval", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--quant", default="4bit")
    p.add_argument("--dtype", default="fp16")
    p.add_argument("--image_mode", default="pad", choices=["pad", "anyres"])


def _add_eval_mcqa(sub):
    p = sub.add_parser("eval-mcqa", help="G1/G5/G7/C3: độ chính xác CSQA/OBQA (+ theo số nút cho C1)")
    p.add_argument("--dataset", choices=["csqa", "obqa"], required=True)
    p.add_argument("--split", required=True, help="csqa: dev; obqa: test")
    p.add_argument("--mode", choices=["text", "image", "kaping", "image+kaping"], required=True)
    p.add_argument("--data_tag", default="", help="vd lowres50 cho G7")
    _model_args(p)

    def run(a):
        from gv.eval.mcqa import eval_mcqa
        print(eval_mcqa(a.dataset, a.split, a.mode, a.adapter, a.exp_id, a.group, a.data_tag, a.max_eval,
                        a.batch_size, a.quant, a.dtype, a.image_mode))

    p.set_defaults(func=run)


def _add_eval_graph_reading(sub):
    p = sub.add_parser("eval-graph-reading", help="G6: 6 tác vụ đọc ảnh đồ thị (Table 4)")
    p.add_argument("--dataset", default="csqa")
    p.add_argument("--split", default="dev")
    _model_args(p)

    def run(a):
        from gv.eval.graph_reading import eval_graph_reading
        print(eval_graph_reading(a.adapter, a.exp_id, a.group, a.dataset, a.split, a.max_eval, a.batch_size,
                                 a.quant, a.dtype, a.image_mode))

    p.set_defaults(func=run)


def _add_eval_vqa(sub):
    p = sub.add_parser("eval-vqa", help="G2/G3: ScienceQA-IMG, MMBench (circular), POPE")
    p.add_argument("--benchmark", choices=["scienceqa", "mmbench", "pope", "all"], required=True)
    _model_args(p)

    def run(a):
        from gv.eval.vqa import eval_vqa
        names = ["scienceqa", "mmbench", "pope"] if a.benchmark == "all" else [a.benchmark]
        for b in names:
            exp = a.exp_id if len(names) == 1 else f"{a.exp_id}-{b}"
            print(b, eval_vqa(b, a.adapter, exp, a.group, a.max_eval, a.batch_size, a.quant, a.dtype, a.image_mode))

    p.set_defaults(func=run)


def _add_resolution_sweep(sub):
    p = sub.add_parser("resolution-sweep", help="C2: độ chính xác CSQA theo độ phân giải ảnh đồ thị")
    p.add_argument("--adapter", default="none")
    p.add_argument("--exp_prefix", required=True, help="vd C2-graphvis hoặc C2-base")
    p.add_argument("--sizes", type=int, nargs="+", default=[50, 112, 224, 336, 672])
    p.add_argument("--image_modes", nargs="+", default=["pad", "anyres"])
    p.add_argument("--max_eval", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--quant", default="4bit")
    p.add_argument("--dtype", default="fp16")

    def run(a):
        from gv.eval.complexity import resolution_sweep
        print(resolution_sweep(a.adapter, a.exp_prefix, a.sizes, a.image_modes, max_eval=a.max_eval,
                               batch_size=a.batch_size, quant=a.quant, dtype=a.dtype))

    p.set_defaults(func=run)


def _add_c1_report(sub):
    p = sub.add_parser("c1-report", help="C1: gộp predictions của các lần eval-mcqa theo số nút đồ thị")
    p.add_argument("--eval_exp_ids", nargs="+", required=True)

    def run(a):
        from gv.eval.complexity import c1_report
        for row in c1_report(a.eval_exp_ids):
            print(row)

    p.set_defaults(func=run)


_EXTRA = [_add_train, _add_eval_mcqa, _add_eval_graph_reading, _add_eval_vqa, _add_resolution_sweep, _add_c1_report]

if __name__ == "__main__":
    main()
