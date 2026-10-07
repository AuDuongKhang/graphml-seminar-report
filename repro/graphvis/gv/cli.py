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


_EXTRA = []   # các task sau đăng ký thêm lệnh (train, eval-*) vào đây

if __name__ == "__main__":
    main()
