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


_EXTRA = [_add_train]   # các task sau đăng ký thêm lệnh eval-* vào đây

if __name__ == "__main__":
    main()
