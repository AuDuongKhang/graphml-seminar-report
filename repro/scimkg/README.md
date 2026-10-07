# Tái lập SciMKG

Hướng dẫn chạy. Kết quả nằm trong báo cáo và trên W&B; khác biệt so với bài báo nằm trong `DEVIATIONS.md`.

## Phần cứng và môi trường

| | Khuyến nghị |
|---|---|
| Nền tảng | Kaggle Notebooks (GPU T4 ×2, bật Internet) hoặc Colab (T4) |
| GPU | Bộ mô hình mở lớn (Qwen3-14B + Gemma3-27B + Phi-3-medium, lượng tử q4 của Ollama) cần ~32 GB → Kaggle T4 ×2. Bộ nhỏ/vừa chạy được trên 1 T4 |
| Phần mềm | Python ≥ 3.10, `pip install -r requirements.txt`, Ollama (`curl -fsSL https://ollama.com/install.sh \| sh`); vLLM cho UniversalNER |
| Tài khoản | W&B (`WANDB_API_KEY`). Tùy chọn: `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `GEMINI_API_KEY` để chạy đúng mô hình của bài |
| Dữ liệu | Zenodo record 19691725: `Groundtruth.zip` + phần cần thiết của `RawData.zip` (tải chọn lọc ~1 GB thay vì 5,4 GB) + `SciMKG.zip` |

## Cách chạy nhanh nhất

Mở `notebooks/SciMKG_Repro.ipynb` trên Kaggle/Colab, sửa ô **CẤU HÌNH**, chạy từ trên xuống.

## Chạy bằng dòng lệnh

```bash
cd repro/scimkg
export GRAPHML_DATA=/path/data GRAPHML_OUT=/path/out WANDB_PROJECT=graphml-seminar-report
python -m smkg.cli --help
```

| Lệnh | Việc |
|---|---|
| `download [--no_images]` | S0: tải nhãn vàng và dữ liệu thô của các bài có nhãn |
| `check-models --models ...` | thử gọi từng mô hình (API hoặc Ollama/vLLM), báo bản thay thế |
| `score-authors` | chấm lại đầu ra LLM gốc của tác giả với nhãn vàng (không gọi mô hình) |
| `evia --models A B C --lessons 30 --run_name S1-...` | chạy EVIA, lưu log phiếu từng (LLM, vòng) |
| `ablate / grid / errors --run_name ... --models ...` | ablation, lưới K × α, phân loại lỗi — tính lại từ log, không gọi mô hình |
| `baseline --name {decomposed_qa,gpt_ner,self_improving,universal_ner} --model ...` | baseline Table 4 |
| `align-eval --metrics clip blip [vqascore]` | điểm căn chỉnh khái niệm–ảnh |
| `human-sheet [--score file.csv]` | tạo / chấm phiếu đánh giá người |
| `regen-align --model ...` | căn chỉnh lại ảnh bằng prompt C1–C4 của tác giả |

Tên mô hình hợp lệ nằm trong `configs/models.yaml`. Mọi lời gọi mô hình được cache tại
`$GRAPHML_OUT/scimkg/cache/`: chạy lại sau khi phiên bị ngắt sẽ bỏ qua bài đã xong và không gọi lại mô hình.

## Kiểm thử

```bash
cd repro && python -m pytest -q
```
