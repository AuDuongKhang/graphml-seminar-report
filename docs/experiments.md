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

Ghi chú tính toán: bài dùng 4×A6000 ~3 giờ (LoRA bf16). Colab T4/L4 16–24 GB có thể cần QLoRA 4-bit,
batch nhỏ + gradient accumulation, hoặc tập con. Ghi rõ mọi sai khác trong VII-A và Phụ lục A
(bảng "Bài báo / Script / Nhóm dùng" đã có sẵn trong `report/sections/99_appendix.tex`).
Các điểm phải quyết định khi dựng code: xem `docs/findings.md` §A3 (lr, batch, pad vs square, CLIP 336, trường `chosen`).

## SciMKG

| ID | Thực nghiệm (bài) | Cấu hình | Mục báo cáo | Trạng thái | W&B |
|---|---|---|---|---|---|
| S0 | Chuẩn bị tập đánh giá + nhãn vàng (bài không công bố → tự gán trên tập con, mô tả quy trình) | Zenodo | XII-A | todo | – |
| S1 | Table 4: EVIA đầy đủ + baseline (Decomposed-QA, GPT-NER, …) | GPT-4o/Gemini-1.5-Flash/DeepSeek-V3 nếu còn API | XII-B | todo | – |
| S2 | Table 4: w/o MLs, w/o SF-FD, w/o cả hai | | XII-B | todo | – |
| S3 | Table 4: bộ LLM mở Qwen3/Gemma3/Phi-3 theo kích thước | Colab/Ollama | XII-B | todo | – |
| S4 | Table 5: lưới iter{1,3,5} × α{0.5,0.7,0.9} | | XII-C | todo | – |
| S5 | Fig 5 phân loại lỗi; Table 7 CLIP/BLIP/X-VLM/VQAScore cho cặp khái niệm–ảnh | | XII-D | todo | – |

Trước khi chạy SciMKG: sửa bước Integration (chuẩn hóa sc/(|M|·K)), dùng đủ 3 LLM ở bước E nếu theo bài,
quyết định có thêm bước chấm tin cậy ảnh >0.8 không — ghi rõ trong XII-A. Chi tiết lỗi: `docs/findings.md` §B2.
Kiểm tra mô hình nào còn được cung cấp qua API (đặc biệt Gemini-1.5) trước khi cam kết cấu hình.

## Quy ước W&B
- Entity/project: `<điền>` — đặt project **public** (hoặc Teams + cross-project report công khai).
- Tên run: `<ID>-<mô tả ngắn>`, ví dụ `G4-joint`, `S2-wo-sffd`.
- Log tối thiểu: config đầy đủ (siêu tham số, seed, commit hash), metric cuối, bảng dự đoán mẫu.
