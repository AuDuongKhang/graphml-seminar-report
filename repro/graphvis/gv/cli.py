"""CLI GraphVis. Chạy từ thư mục repro/graphvis: python -m gv.cli <lệnh> ...  (xem README.md)."""
import argparse
import sys
from pathlib import Path

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

    def run(a):
        from gv.pipeline import build_data
        out = build_data(a.dataset, a.split, a.scale, a.image_size, a.tag, a.max_nodes, not a.no_wandb)
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


_EXTRA = [_add_train, _add_eval_mcqa, _add_eval_graph_reading]   # Task 8 thêm eval-vqa, resolution-sweep

if __name__ == "__main__":
    main()
