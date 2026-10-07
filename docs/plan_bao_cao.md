# Kế hoạch & sườn báo cáo giữa kỳ — GraphVis (seminar) + SciMKG (chỉ báo cáo)

## 0. Ràng buộc (từ `requirement_report.txt`)

| Ràng buộc | Cách đáp ứng |
|---|---|
| 1 file PDF `MSHV1-MSHV2.pdf`, < 20 MB, **≤ 25 trang** (gồm bìa, tham khảo, phụ lục) | Mục tiêu 20–22 trang, chừa 3 trang dự phòng; ảnh nén PNG/PDF vector |
| Template IEEE (`report/conference_101719.tex`) | Giữ `IEEEtran` conference, thêm trang bìa 1 cột trước `\maketitle` |
| Hai bài là **hai phần của một báo cáo** | Mở đầu chung + khung liên kết + thảo luận tổng hợp + kết luận chung |
| Không sao chép câu chữ bài gốc | Viết lại bằng lời nhóm; vẽ lại hình (TikZ / draw.io) thay vì chụp hình bài |
| Tái lập → log W&B công khai, gắn link **đúng từng mục** (kết quả chính, ablation, …) | Mỗi mục thực nghiệm có 1 W&B report link riêng |
| Code trên GitHub, README chỉ hướng dẫn chạy (môi trường, phần cứng, lệnh) | 1 repo, 2 thư mục `graphvis/`, `scimkg/`; link trong mục tái lập |
| Một ngôn ngữ thống nhất | (cần chốt — đề xuất tiếng Việt, biên dịch XeLaTeX) |

## 1. Nhận xét của thầy và cách đưa vào báo cáo

### Nhận xét 1 — [Slide 3] "Mức độ hình ảnh là bao nhiêu thì mô hình trả lời sai? (thông số kích thước, …)"
Slide 3 nói ConceptNet có 799.273 nút / 2.487.810 cạnh nên phải cắt đồ thị con. Câu hỏi: đồ thị con/ảnh phức tạp tới mức nào thì mô hình bắt đầu đọc sai.

**Bài báo trả lời được gì:**
- Chỉ dùng đồ thị con **2-hop** "để ảnh không quá rối" (Sec 5.1, Limitations) — không có ngưỡng định lượng.
- Phân bố kích thước đồ thị trên CSQA: số nút tới ~80, số cạnh tới ~160 (Fig 4–5).
- Ảnh độ phân giải thấp **50×50** làm giảm độ chính xác CSQA (Fig 9, phụ lục) — chỉ thử 1 mức.
- Đọc hình vẫn rất yếu sau fine-tune: đếm nút 27,5%, liệt kê triple 8,2% (Table 4) — không tách theo kích thước đồ thị.
- Bối cảnh kỹ thuật (nhóm bổ sung): LLaVA-1.6 dùng CLIP ViT-L/14-336 + AnyRes (lưới tối đa ~672×672), mỗi ô 336 px → 576 token; nhãn nút chữ nhỏ trong ảnh lớn bị nén mạnh.

**Bài báo KHÔNG trả lời** → nhóm tự làm thực nghiệm bổ sung (mục 7.3 dưới):
1. Accuracy (QA + 6 tác vụ đọc hình) theo **bin số nút / số cạnh** (vd ≤10, 11–20, 21–40, >40).
2. Accuracy theo **độ phân giải ảnh** (50, 112, 224, 336, 672 px) và/hoặc DPI Graphviz.
3. Kết luận dưới dạng "ngưỡng" quan sát được + đề xuất (giới hạn số nút, tách ảnh, layout).

### Nhận xét 2 — [Slide 11] "Cách huấn luyện mô hình ở slide này?"
Slide 11 chỉ vẽ "giai đoạn 1 → mang tham số → giai đoạn 2", chưa nói rõ huấn luyện cái gì, như thế nào.

**Nội dung bổ sung (mục 4.5 dưới), lấy từ bài + code `GraphVis-code`:**
- Thành phần nào được cập nhật: CLIP ViT-L/14 **đóng băng**; projector MLP 2 lớp (mlp2x_gelu) **được huấn luyện** (lr 2e-5); LLM Mistral-7B **đóng băng + LoRA** (r=128, α=256, gắn vào mọi lớp tuyến tính).
- Hàm mất mát: NLL/cross-entropy **chỉ trên token câu trả lời** (phần prompt bị mask bằng `IGNORE_INDEX` trong `llava/train/train.py`).
- Lịch: giai đoạn 1 trên D_g (6 tác vụ, 1 epoch) → khởi tạo giai đoạn 2 từ trọng số giai đoạn 1 → D_v (1 epoch). AdamW, cosine, warmup 3%, bf16, DeepSpeed ZeRO-2, max length 2048, ảnh `pad`.
- Tài nguyên: 4×RTX A6000, ~3 giờ.
- **Điểm không khớp nhóm phát hiện**: bài ghi LR 1e-7 & global batch 4; script ghi LR 2e-5, batch 16/GPU × 2 GPU; README chỉ có 1 lệnh train (không tách 2 giai đoạn); code vẽ Graphviz và code sinh D_v không được công bố.
- Hình vẽ lại: sơ đồ khối LLaVA đánh dấu khối đóng băng / LoRA / huấn luyện đầy đủ + timeline 2 giai đoạn.

## 2. Sườn báo cáo (dự kiến số trang, 2 cột IEEE)

| # | Mục | Trang |
|---|---|---|
| — | **Trang bìa** (trường, môn, GVHD, nhóm, MSHV, 2 bài báo) | 1 |
| — | Tóm tắt + Từ khóa | 0,3 |
| I | **Giới thiệu chung** — bối cảnh LLM ↔ KG; khung liên kết 2 bài: *LLM → KG* (SciMKG dùng LLM/MLLM để **xây** KG đa phương thức) và *KG → LLM* (GraphVis dùng KG dạng ảnh để **tăng cường** LVLM); đóng góp của báo cáo; cấu trúc | 1 |
| II | **Kiến thức nền** — KG & ConceptNet; LLM/LVLM (LLaVA: vision encoder + projector + LLM); SFT/NLL; LoRA; curriculum learning; NER bằng LLM, Self-Refine, Self-Consistency; chỉ số CLIP/BLIP score | 1,5 |
| | **PHẦN A — GraphVis (NeurIPS 2024)** | |
| III | Bài toán & động cơ — hình thức hóa f (truy hồi), g (đưa vào prompt); hai hướng cũ (linearize/KAPING, GNN/GNP) và đánh đổi | 1 |
| IV | Phương pháp — 4.1 truy hồi 2-hop & Graphviz; 4.2 luồng ảnh qua LVLM; 4.3 GĐ1: 6 tác vụ đọc hình (OCR vs Graph); 4.4 GĐ2: QA có ảnh, prompt; **4.5 Chi tiết huấn luyện (trả lời NX slide 11)** | 2,5 |
| V | Kết quả công bố — Table 1 (CSQA/OBQA), Table 2 (ScienceQA/MMBench/POPE), ablation curriculum vs joint, KAPING, Table 4 đọc hình — kèm bình luận cách đọc số | 1,5 |
| VI | **Ảnh hưởng độ phức tạp ảnh (trả lời NX slide 3)** — tổng hợp bằng chứng trong bài + phân tích của nhóm | 1 |
| VII | Tái lập — 7.1 thiết lập & khác biệt so với bài; 7.2 kết quả chính [W&B]; 7.3 ablation curriculum vs joint [W&B]; 7.4 thực nghiệm độ phức tạp ảnh [W&B] | 2,5 |
| VIII | Đánh giá phê bình — ưu điểm; hạn chế (1 backbone, 1 KG, không có độ lệch chuẩn, so sánh khác backbone, code thiếu/lỗi: prompt thiếu dấu phẩy, `{node\"}`, đếm cạnh theo triple duy nhất nhưng bậc theo link, gộp 4 đồ thị/câu hỏi) | 0,8 |
| | **PHẦN B — SciMKG (AAAI 2026)** | |
| IX | Bài toán & động cơ — EKG phụ thuộc gán nhãn, thiếu đa phương thức; so sánh Table 1 | 0,7 |
| X | Phương pháp — thu thập/tiền xử lý; pipeline EVIA (Extraction, Verification/Self-Refine, Integration/SC-score với τ, Augmentation ConceptNet 8 quan hệ + Wikipedia); căn chỉnh đa phương thức (ảnh: MLLM + ngưỡng 0,8; video: timestamp; audio: TTS); tổ chức 3 tầng, 11 quan hệ, JSON + RDF | 2 |
| XI | Kết quả công bố — Table 4 (so sánh, ablation, cỡ LLM), Table 5 (iter × τ), lỗi (Fig 5), Table 7 căn chỉnh | 1 |
| XII | Tái lập — trích khái niệm trên tập con bằng LLM mã nguồn mở / API; ablation w/o MLs, w/o SF-FD; quét iter × τ [W&B] | 1,5 |
| XIII | Đánh giá phê bình — ACC/AUC cao do mất cân bằng lớp; MCC của phương pháp (0,729) **thấp hơn** Decomposed-QA (0,741) dù bài nói vượt mọi baseline; chênh F1 ghi 0,062 vs bảng 0,064; Python 3.7 + PyTorch 2.4 không tương thích; video không được đánh giá; audio = TTS của text; ảnh chỉ phủ 39% khái niệm; nhãn vàng tập đánh giá không mô tả | 0,8 |
| XIV | **Thảo luận tổng hợp** — hai chiều LLM↔KG; cả hai dùng ConceptNet; ý tưởng nối: dùng đồ thị con SciMKG vẽ kiểu GraphVis cho VQA khoa học (ScienceQA) | 0,5 |
| XV | Kết luận + hướng phát triển | 0,3 |
| — | Phân công công việc (bảng ngắn), Tài liệu tham khảo | 1 |
| — | Phụ lục: siêu tham số, prompt, link W&B/GitHub | 0,5–1 |
| | **Tổng** | **≈ 21–22** |

## 3. Hình/bảng dự kiến (vẽ lại, không chụp)
1. Hình khung liên kết 2 bài (LLM→KG / KG→LLM).
2. GraphVis: pipeline dữ liệu + 2 giai đoạn (vẽ lại slide 11, bổ sung khối đóng băng/LoRA).
3. GraphVis: ví dụ 1 ảnh đồ thị con + 6 câu hỏi D_g (rút gọn).
4. Biểu đồ accuracy theo số nút/độ phân giải (kết quả nhóm).
5. SciMKG: pipeline EVIA + căn chỉnh đa phương thức.
6. Bảng: kết quả công bố vs tái lập (mỗi bài 1–2 bảng).

## 4. Kế hoạch thực hiện

| Bước | Việc | Đầu ra |
|---|---|---|
| 1 | Chốt ngôn ngữ, thành viên/MSHV, phạm vi tái lập & tài nguyên GPU/API | — |
| 2 | Dựng khung LaTeX: bìa, các section rỗng, gói tiếng Việt (XeLaTeX), bib | `main.tex` biên dịch được |
| 3 | Viết Phần A (III–VI, VIII) + 2 mục trả lời nhận xét | ~8 trang |
| 4 | Viết Phần B (IX–XI, XIII) | ~4,5 trang |
| 5 | Repo GitHub + tái lập GraphVis (data → GĐ1 → GĐ2 → eval) log W&B | link W&B, bảng số |
| 6 | Thực nghiệm độ phức tạp ảnh (NX slide 3) | biểu đồ + bảng |
| 7 | Tái lập SciMKG trên tập con, log W&B | bảng số |
| 8 | Viết VII, XII, I, XIV, XV, tóm tắt; vẽ hình | bản đầy đủ |
| 9 | Kiểm tra: ≤25 trang, <20 MB, không lẫn ngôn ngữ, link W&B công khai, đạo văn | `MSHV1-MSHV2.pdf` |

Ghi chú tài nguyên: máy hiện tại có RTX 3060 Laptop 6 GB — không đủ fine-tune LLaVA-1.6-7B; cần Colab/Kaggle (T4 16 GB, QLoRA 4-bit, tập con) hoặc thuê A100/A6000.

## 5. Quyết định đã chốt (07/10/2026)
- Ngôn ngữ: **tiếng Việt**, biên dịch XeLaTeX (`report/`, `latexmk -xelatex main`).
- GraphVis: dùng **Colab**, mục tiêu **tái lập tất cả thí nghiệm**, log W&B công khai.
- SciMKG: ưu tiên **đúng cấu hình bài** (GPT-4o, Gemini-1.5-Flash, DeepSeek-V3; MLLM GPT-4o/Gemini-1.5-Pro); không được thì ghi rõ và dùng LLM mở.

## 6. Danh sách thực nghiệm cần tái lập (mỗi dòng = 1 link W&B)

| ID | Bài | Thực nghiệm | Mục báo cáo |
|---|---|---|---|
| G1 | GraphVis | Table 1: CSQA/OBQA — LVLM gốc, KAPING, GraphVis | VII-B |
| G2 | GraphVis | Table 2: ScienceQA-IMG, MMBench, POPE ×3 | VII-C |
| G3 | GraphVis | Table 6: ScienceQA joint vs curriculum | VII-C |
| G4 | GraphVis | Fig 7: joint / curriculum / graph-first / OCR-first | VII-D |
| G5 | GraphVis | Table 3: ±KAPING trên base / joint / curriculum | VII-D |
| G6 | GraphVis | Table 4: 6 tác vụ đọc hình trước/sau GĐ1 | VII-E |
| G7 | GraphVis | Fig 9: ảnh 50×50 | VII-E |
| C1–C3 | GraphVis (mới) | NX slide 3: theo số nút/cạnh, theo độ phân giải, đối chứng không ảnh | VI-C |
| S1 | SciMKG | Table 4: EVIA đầy đủ + baseline | XII-B |
| S2 | SciMKG | Table 4: ablation w/o MLs, w/o SF-FD, cả hai | XII-B |
| S3 | SciMKG | Table 4: bộ LLM mở theo kích thước (Qwen3/Gemma3/Phi-3) | XII-B |
| S4 | SciMKG | Table 5: lưới iter × α | XII-C |
| S5 | SciMKG | Fig 5: phân loại lỗi; Table 7: CLIP/BLIP/X-VLM/VQAScore | XII-D |

## 6b. Điều chỉnh theo góp ý (07/10/2026)
- Tiêu đề: "Báo cáo hai mô hình: GraphVis và SciMKG"; bìa có logo, GV Lê Ngọc Thành, Âu Dương Khang (25C15045), Nguyễn Bùi Mẫn Nhi (25C15055), tháng 10/2026.
- Không ghi "trả lời nhận xét của thầy": nội dung lồng vào IV-E (Chi tiết huấn luyện) và VI (Độ phức tạp ảnh đồ thị) như thông tin chi tiết hơn.
- Bỏ khung liên kết hai bài (Hình framework, mục Thảo luận tổng hợp); chỉ còn Kết luận.
- Phần SciMKG viết chi tiết theo mã nguồn: bảng mô hình từng bước, tiền xử lý 5 bước, mã giả EVIA, ví dụ bỏ phiếu, prompt ảnh C1–C4, căn chỉnh video theo mốc thời gian, schema JSON, sinh bộ ba, RDF, điểm không khớp bài–mã.

## 7. Trạng thái
- [x] Khung LaTeX, bìa, hình TikZ, bib — biên dịch sạch (10 trang hiện tại).
- [x] Phần A (III–VI, VIII) và Phần B (IX–XI, XIII) viết xong bản nháp, gồm 2 mục trả lời nhận xét.
- [ ] I, II, XIV, XV, tóm tắt (đang là sườn `\todo`).
- [ ] VII, XII + VI-C (chờ thực nghiệm). Xóa mọi `\todo` trước khi nộp.
