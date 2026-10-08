# Danh sách thực nghiệm tái lập

Mỗi dòng = một W&B run/report công khai, gắn vào đúng mục của báo cáo bằng `\wandb{url}`.
Cập nhật cột *Trạng thái* và *W&B* ngay khi xong. Không ghi số liệu vào README (chỉ vào báo cáo).

Trạng thái: `todo` · `đang chạy` · `xong` · `bỏ (lý do)`

## GraphVis (Kaggle/Colab GPU miễn phí)

Chạy bằng `repro/graphvis/notebooks/GraphVis_Repro.ipynb` — mỗi ID có một ô. Sai khác: `repro/graphvis/DEVIATIONS.md`.

| ID | Thực nghiệm (bài) | Cấu hình | Mục báo cáo | Trạng thái | W&B |
|---|---|---|---|---|---|
| G0 | Dựng dữ liệu: truy hồi 2-hop, vẽ Graphviz, sinh D_g (6 tác vụ) và D_v | CSQA, OBQA | VII-A | code sẵn sàng | – |
| G1 | Table 1: CSQA/OBQA — base, KAPING, GraphVis | | VII-B | code sẵn sàng | – |
| G2 | Table 2: ScienceQA-IMG, MMBench, POPE ×3 | eval LLaVA | VII-C | code sẵn sàng | – |
| G3 | Table 6: ScienceQA joint vs curriculum | | VII-C | code sẵn sàng | – |
| G4 | Fig 7: joint / curriculum / graph-first / OCR-first | CSQA | VII-D | code sẵn sàng | – |
| G5 | Table 3: ±KAPING trên base / joint / curriculum | CSQA | VII-D | code sẵn sàng | – |
| G6 | Table 4: 6 tác vụ đọc hình trước/sau GĐ1 (exact match) | test CSQA | VII-E | code sẵn sàng | – |
| G7 | Fig 9: ảnh 50×50 | CSQA | VII-E | code sẵn sàng | – |
| C1 | Độ chính xác theo số nút/cạnh (≤10, 11–20, 21–40, >40) | base, GraphVis, C3 | VI-C | code sẵn sàng | – |
| C2 | Độ chính xác theo độ phân giải (50/112/224/336/672, AnyRes) | | VI-C | code sẵn sàng | – |
| C3 | Đối chứng: tinh chỉnh CSQA **không ảnh**, cùng siêu tham số | | VI-C | code sẵn sàng | – |

Ghi chú tính toán: bài dùng 4×A6000 ~3 giờ (LoRA bf16). Code mặc định QLoRA 4-bit fp16, 1 GPU, quy mô `lite`
(xem `repro/graphvis/README.md`). Mọi sai khác ghi trong `repro/graphvis/DEVIATIONS.md` → đưa vào VII-A và Phụ lục A
(bảng "Bài báo / Script / Nhóm dùng" trong `report/sections/99_appendix.tex`, cột "Nhóm dùng" điền theo config đã chạy).

## SciMKG (Kaggle/Colab, mô hình mở qua Ollama nếu không có API key)

Chạy bằng `repro/scimkg/notebooks/SciMKG_Repro.ipynb`. Sai khác: `repro/scimkg/DEVIATIONS.md`.

| ID | Thực nghiệm (bài) | Cấu hình | Mục báo cáo | Trạng thái | W&B |
|---|---|---|---|---|---|
| S0 | Chuẩn bị tập đánh giá + nhãn vàng (bài không công bố → tự gán trên tập con, mô tả quy trình) | Zenodo | XII-A | code sẵn sàng | – |
| S1 | Table 4: EVIA đầy đủ + baseline (Decomposed-QA, GPT-NER, Self-Improving, UniversalNER) + chấm lại đầu ra gốc của tác giả (`score-authors`, không tốn API) | GPT-4o/Gemini-1.5-Flash/DeepSeek-V3 nếu còn API | XII-B | code sẵn sàng | – |
| S2 | Table 4: w/o MLs, w/o SF-FD, w/o cả hai | | XII-B | code sẵn sàng | – |
| S3 | Table 4: bộ LLM mở Qwen3/Gemma3/Phi-3 theo kích thước | Colab/Ollama | XII-B | code sẵn sàng | – |
| S4 | Table 5: lưới iter{1,3,5} × α{0.5,0.7,0.9} | | XII-C | code sẵn sàng | – |
| S5 | Fig 5 phân loại lỗi; Table 7 CLIP/BLIP/X-VLM/VQAScore cho cặp khái niệm–ảnh | | XII-D | code sẵn sàng | – |

Đã xử lý trong code: tích hợp chuẩn hóa theo số (LLM, vòng) chạy thành công; cả 3 LLM trích xuất (`--extractors`
để chạy như mã gốc); `regen-align` theo mã gốc (không có bước chấm tin cậy 0,8) — ghi rõ trong XII-A theo
`repro/scimkg/DEVIATIONS.md`. Trước khi chạy: `check-models` để biết mô hình nào dùng được; mô hình có `fallback`
được tự thay và tên mô hình thực tế được ghi trong W&B (`resolved_models`).

## Quy ước W&B
- Entity/project: `<điền>` — đặt project **public** (hoặc Teams + cross-project report công khai).
- Tên run: `<ID>-<mô tả ngắn>`, ví dụ `G4-joint`, `S2-wo-sffd`.
- Log tối thiểu: config đầy đủ (siêu tham số, seed, commit hash), metric cuối, bảng dự đoán mẫu.
