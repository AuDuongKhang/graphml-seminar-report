# Tái lập GraphVis

Hướng dẫn chạy. Kết quả nằm trong báo cáo và trên W&B; khác biệt so với bài báo nằm trong `DEVIATIONS.md`.

## Phần cứng và môi trường

| | Khuyến nghị |
|---|---|
| Nền tảng | Kaggle Notebooks (GPU T4 ×2 hoặc P100, bật Internet) hoặc Google Colab (GPU T4) |
| GPU | ≥ 15 GB VRAM. Mặc định QLoRA 4-bit + fp16; với A100/L4 dùng `quant=none dtype=bf16` |
| RAM | ≥ 12 GB (dựng ConceptNet chiếm ~0,6 GB) |
| Đĩa | ~15 GB cache mô hình (HF Hub) + < 2 GB dữ liệu ở `lite` + trong lúc train ~4 GB/checkpoint (bị xóa khi xong giai đoạn) + ~0,7 GB/adapter. Kaggle: bật *Persistence → Files only*; nên dùng `hub_repo` để lưu adapter |
| Phần mềm | Python ≥ 3.10, Graphviz (`apt-get install graphviz`), `pip install -r requirements.txt` |
| Tài khoản | W&B (`WANDB_API_KEY`); tùy chọn HF Hub (`HF_TOKEN`) để lưu adapter; `GITHUB_TOKEN` (fine-grained, *Contents: Read-only*) nếu repo còn private |

## Cách chạy nhanh nhất

Mở `notebooks/GraphVis_Repro.ipynb` trên Kaggle/Colab, sửa ô **CẤU HÌNH**, chạy từ trên xuống.
Notebook tự clone repo, cài thư viện, đặt dữ liệu vào Drive (Colab) hoặc `/kaggle/working` (Kaggle).

## Chạy bằng dòng lệnh

```bash
cd repro/graphvis
export GRAPHML_DATA=/path/data GRAPHML_OUT=/path/out WANDB_PROJECT=graphml-seminar-report
python -m gv.cli --help
```

| Lệnh | Việc |
|---|---|
| `build-data --dataset csqa --split train --scale lite` | G0: tải dữ liệu QA-GNN (~60 MB), truy hồi đồ thị con, vẽ ảnh, sinh `dg/dv/text.jsonl` |
| `train --config configs/<file>.yaml [--set k=v ...] [--dry_run]` | huấn luyện các giai đoạn trong config |
| `eval-mcqa --dataset csqa --split dev --mode {text,image,kaping,image+kaping} --adapter <dir> --exp_id <tên>` | độ chính xác CSQA/OBQA |
| `eval-graph-reading --adapter <dir> --exp_id <tên>` | 6 tác vụ đọc ảnh đồ thị |
| `eval-vqa --benchmark {scienceqa,mmbench,pope,all} --adapter <dir> --exp_id <tên>` | VQA zero-shot |
| `resolution-sweep --adapter <dir> --exp_prefix <tên>` | độ chính xác theo độ phân giải ảnh |
| `c1-report --eval_exp_ids <tên> ...` | gộp độ chính xác theo số nút |

Adapter của một giai đoạn nằm ở `$GRAPHML_OUT/graphvis/<exp_id>/<stage>/final`. `--adapter none` = mô hình gốc.
Thứ tự và phụ thuộc giữa các thực nghiệm: xem bảng đầu notebook hoặc `docs/experiments.md`.

## Quy mô (`--scale`)

| | tiny | lite (mặc định) | full |
|---|---|---|---|
| D_g (ảnh × 6 tác vụ) | 40 ảnh | 600 ảnh | toàn bộ tập train |
| D_v | 200 câu | 2.000 câu | toàn bộ |
| Đánh giá CSQA dev / OBQA test | 100 câu | đủ | đủ |

## Chạy lại sau khi phiên bị ngắt

`train --dry_run` ghi vào `<exp_id>-dryrun/`, không ảnh hưởng run thật.

Chạy lại đúng ô/lệnh: dữ liệu đã có sẽ được dùng lại; giai đoạn có `final/` bị bỏ qua; giai đoạn dở dang
resume từ `checkpoint-*` gần nhất (lưu mỗi 500 bước); W&B run tiếp tục cùng id.

## Kiểm thử

```bash
cd repro && python -m pytest -q            # CPU, không cần mạng
python -m pytest -q -m gpu graphvis/tests  # trên máy có GPU
```
