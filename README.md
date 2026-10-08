# graphml-seminar-report

Mã nguồn tái lập thực nghiệm cho báo cáo giữa kỳ môn **Học máy trên đồ thị** (ĐH KHTN – ĐHQG-HCM):
**GraphVis** (NeurIPS 2024) và **SciMKG** (AAAI 2026).
Nhóm: Âu Dương Khang (25C15045), Nguyễn Bùi Mẫn Nhi (25C15055).

README này chỉ mô tả **cách chạy lại**. Kết quả và phân tích nằm trong báo cáo PDF.

## Lấy mã nguồn

```bash
git clone --recurse-submodules https://github.com/AuDuongKhang/graphml-seminar-report.git
cd graphml-seminar-report
# nếu đã clone mà thiếu submodule:
git submodule update --init
```

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `report/` | Mã LaTeX của báo cáo (XeLaTeX) |
| `GraphVis-code/` | Mã gốc GraphVis (Apache-2.0) và các chỉnh sửa của nhóm để tái lập |
| `SciMKG/` | Submodule mã gốc SciMKG (`kg-bnu/SciMKG`) |
| `repro/` | Code tái lập của nhóm: `graphvis/` (package `gv`), `scimkg/` (package `smkg`), `common/` |
| `docs/` | Tài liệu nội bộ cho nhóm |

## Biên dịch báo cáo

Yêu cầu: TeX Live hoặc MiKTeX có `xelatex`, `bibtex`, `latexmk`; font Times New Roman.

```bash
cd report
latexmk -xelatex main
```

## Tái lập GraphVis

Chi tiết: [`repro/graphvis/README.md`](repro/graphvis/README.md).

- **Phần cứng:** Kaggle Notebooks (T4 ×2 / P100) hoặc Google Colab (T4), GPU ≥ 15 GB; mặc định QLoRA 4-bit.
- **Môi trường:** Python ≥ 3.10, Graphviz, `pip install -r repro/graphvis/requirements.txt`.
- **Chạy:** mở `repro/graphvis/notebooks/GraphVis_Repro.ipynb`, sửa ô cấu hình, chạy từ trên xuống
  (mỗi ô ứng với một ID thực nghiệm G0–G7, C1–C3).
- **Dòng lệnh:** `cd repro/graphvis && python -m gv.cli --help`.

## Tái lập SciMKG

Chi tiết: [`repro/scimkg/README.md`](repro/scimkg/README.md).

- **Phần cứng:** Kaggle Notebooks (T4 ×2 khuyến nghị) hoặc Colab (T4); mô hình mở chạy bằng Ollama.
- **Môi trường:** Python ≥ 3.10, `pip install -r repro/scimkg/requirements.txt`, Ollama; vLLM cho baseline UniversalNER.
- **Khóa API (tùy chọn):** `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `GEMINI_API_KEY` — đặt trong Secrets của notebook, không commit.
- **Chạy:** mở `repro/scimkg/notebooks/SciMKG_Repro.ipynb`, sửa ô cấu hình, chạy từ trên xuống (S0–S5).
- **Dòng lệnh:** `cd repro/scimkg && python -m smkg.cli --help`.

## Kiểm thử

```bash
cd repro && python -m pytest -q     # chạy trên CPU, không cần mạng
```

## Theo dõi thực nghiệm

Mọi lần chạy được ghi bằng Weights & Biases (`wandb login` trước khi chạy).
Link công khai của từng thực nghiệm được dẫn trong báo cáo.
