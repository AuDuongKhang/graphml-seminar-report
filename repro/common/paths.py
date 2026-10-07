"""Thư mục dữ liệu/kết quả. Đặt qua biến môi trường để notebook Colab/Kaggle trỏ ra Drive hoặc /kaggle/working."""
import os
from pathlib import Path


def data_root() -> Path:
    p = Path(os.environ.get("GRAPHML_DATA", "_data"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def out_root() -> Path:
    p = Path(os.environ.get("GRAPHML_OUT", "_out"))
    p.mkdir(parents=True, exist_ok=True)
    return p
