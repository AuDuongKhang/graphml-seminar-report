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
| `repro/` | Notebook/script tái lập của nhóm *(đang xây dựng)* |
| `docs/` | Tài liệu nội bộ cho nhóm |

## Biên dịch báo cáo

Yêu cầu: TeX Live hoặc MiKTeX có `xelatex`, `bibtex`, `latexmk`; font Times New Roman.

```bash
cd report
latexmk -xelatex main
```

## Tái lập GraphVis

> *Đang xây dựng.* Sẽ bổ sung: môi trường (Python, CUDA, thư viện), phần cứng (Google Colab, loại GPU),
> tải dữ liệu (CSQA, OBQA, ConceptNet, ScienceQA, MMBench, POPE), lệnh sinh dữ liệu, lệnh huấn luyện
> giai đoạn 1 và 2, lệnh đánh giá, và notebook tương ứng với từng thực nghiệm.

## Tái lập SciMKG

> *Đang xây dựng.* Sẽ bổ sung: môi trường, khóa API cần có (đặt trong `.env`, không commit), tải dữ liệu,
> lệnh chạy chuỗi trích xuất khái niệm, các ablation, và lệnh đánh giá.

## Theo dõi thực nghiệm

Mọi lần chạy được ghi bằng Weights & Biases (`wandb login` trước khi chạy).
Link công khai của từng thực nghiệm được dẫn trong báo cáo.
