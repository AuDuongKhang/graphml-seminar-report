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
| `build-data --dataset csqa --split train --scale lite [--hub_repo <repo>]` | G0: tải dữ liệu QA-GNN (~60 MB), truy hồi đồ thị con, vẽ ảnh, sinh `dg/dv/text.jsonl`; có `--hub_repo` (hoặc `GRAPHML_HUB_REPO`) thì lấy/lưu bản đã dựng trên HF Hub |
| `train --config configs/<file>.yaml [--set k=v ...] [--dry_run]` | huấn luyện các giai đoạn trong config |
| `eval-mcqa --dataset csqa --split dev --mode {text,image,kaping,image+kaping} --adapter <dir> --exp_id <tên>` | độ chính xác CSQA/OBQA |
| `eval-graph-reading --adapter <dir> --exp_id <tên>` | 6 tác vụ đọc ảnh đồ thị |
| `eval-vqa --benchmark {scienceqa,mmbench,pope,all} --adapter <dir> --exp_id <tên>` | VQA zero-shot |
| `resolution-sweep --adapter <dir> --exp_prefix <tên>` | độ chính xác theo độ phân giải ảnh |
| `c1-report --eval_exp_ids <tên> ...` | gộp độ chính xác theo số nút |
| `python -X utf8 scripts/analyze_predictions.py <thư mục>` | đọc các bảng `predictions` xuất CSV từ W&B: McNemar, theo số nút, MMBench theo nhóm câu, G6 theo tác vụ |

Adapter của một giai đoạn nằm ở `$GRAPHML_OUT/graphvis/<exp_id>/<stage>/final`. `--adapter none` = mô hình gốc.
Thứ tự và phụ thuộc giữa các thực nghiệm: xem bảng đầu notebook hoặc `docs/experiments.md`.

## Quy mô (`--scale`)

| | tiny | lite (mặc định) | full |
|---|---|---|---|
| D_g (ảnh × 6 tác vụ) | 40 ảnh | 600 ảnh | toàn bộ tập train |
| D_v | 200 câu | 2.000 câu | toàn bộ |
| Đánh giá CSQA dev / OBQA test | 100 câu | đủ | đủ |

## Chạy theo lượt trên Kaggle (Save Version)

Toàn bộ thực nghiệm cần nhiều hơn giới hạn 12 giờ của một lượt chạy Kaggle, nên notebook chia thành 9 lượt
(bảng `BATCHES` trong ô CẤU HÌNH). Mỗi lượt: đặt `BATCH = <số lượt>` → *Save Version → Save & Run All* → đóng trình duyệt.
Ô không thuộc lượt tự bỏ qua; lệnh lỗi làm dừng cả lượt. Chạy các lượt theo thứ tự 1 → 9. `BATCH = 0` chạy tất cả (chỉ cho phiên tương tác).

Mỗi lần *Save Version* bắt đầu với `/kaggle/working` trống, nên lượt 2–9 **bắt buộc** đặt `HUB_REPO` (secret `HF_TOKEN` quyền Write):
- dữ liệu G0: `build-data` đẩy mỗi split thành `data/<scale>/<dataset>/<split>.tar` lên repo đó; lượt sau tải về thay vì dựng lại
  (lần đầu thiếu bản trên Hub thì dựng lại, seed cố định nên giống hệt);
- adapter: `<exp>/<stage>/` trên cùng repo, tự tải khi train/eval cần;
- dự đoán cho C1: thiếu `predictions.jsonl` thì đọc bảng `predictions` của run W&B cùng tên.

## Chạy lại sau khi phiên bị ngắt

`train --dry_run` ghi vào `<exp_id>-dryrun/`, không ảnh hưởng run thật.

Chạy lại đúng ô/lệnh: dữ liệu đã có sẽ được dùng lại; giai đoạn có `final/` bị bỏ qua; giai đoạn dở dang
resume từ `checkpoint-*` gần nhất (lưu mỗi 500 bước); W&B run tiếp tục cùng id.

## Kiểm thử

```bash
cd repro && python -m pytest -q            # CPU, không cần mạng
python -m pytest -q -m gpu graphvis/tests  # trên máy có GPU
```
