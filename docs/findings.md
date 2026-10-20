# Những gì đã kiểm chứng về GraphVis (bài báo ↔ mã nguồn)

Số dòng tính theo `GraphVis-code/` ở commit đầu. Phát hiện về SciMKG nằm riêng trong [SciMKG/docs/FINDINGS.md](../SciMKG/docs/FINDINGS.md).

---

## A. GraphVis

### A1. Thiết lập (bài báo)
- Backbone `llava-v1.6-mistral-7b`: CLIP ViT-L/14-336 (đóng băng) → projector `mlp2x_gelu` (học toàn bộ, lr 2e-5)
  → Mistral-7B (đóng băng + LoRA r=128, α=256, mọi lớp tuyến tính).
- KG: ConceptNet, 799.273 nút, 2.487.810 cạnh, 42 quan hệ gộp còn 17. Truy hồi đường đi ≤2 bước (theo MHGRN, Feng 2020).
  Vẽ bằng Graphviz.
- Giai đoạn 1 (D_g): 6 tác vụ, mỗi tác vụ 5 mẫu câu — tên nút, bậc nút, nút bậc cao nhất, số nút, số cạnh, liệt kê bộ ba.
  Nhóm OCR = tên nút + bộ ba; nhóm Graph = 4 tác vụ còn lại.
- Giai đoạn 2 (D_v): prompt "The image represents a knowledge graph relevant to the question, which may or may not be useful. Question: ...", đáp án gốc.
- Bảng 5 bài (Phụ lục A): lr **1e-7**, global batch **4**, AdamW, cosine, warmup 0.03, wd 0, max len 2048,
  `image_aspect_ratio=pad`, `group_by_modality_length=True`, 1 epoch mỗi giai đoạn. 4×RTX A6000 ~3 giờ. Greedy decoding.
- Dữ liệu đánh giá: CSQA, OBQA (QA văn bản); ScienceQA-IMG, MMBench, POPE (VQA, zero-shot, script đánh giá của LLaVA).

### A2. Số liệu công bố (đã đưa vào báo cáo)
| Bảng/Hình bài | Nội dung |
|---|---|
| Table 1 | Base LVLM 70.5/75.6; KAPING 67.7/71.2; GraphVis 82.8/85.5 (CSQA/OBQA). QA-GNN 76.1/82.8; GreaseLM 78.5/84.8; KSL GPT-3.5 79.6/81.6; GNP FLAN-T5-11B –/79.8 |
| Table 2 | SciQA-IMG 68.86→73.18; MMBench 63.75→66.41; POPE F1 rand 87.65→89.12, pop 86.53→87.89, adv 85.37→86.32 (Acc 88.56→89.73, 87.73→88.73, 86.47→87.07) |
| Fig 7 | Base 70.52; Joint 78.05; Curriculum 82.56; Graph-first 81.00; OCR-first 82.80 |
| Table 3 | +KAPING: base 67.65; joint 78.71; curriculum 81.74 |
| Table 4 | Đọc hình trước→sau: tên nút 1.4→12.8; bậc 15.3→27.0; bậc max 3.3→11.6; số nút 16.7→27.5; số cạnh 9.7→16.2; bộ ba 0.6→8.2 |
| Table 6 | ScienceQA: joint 71.94, curriculum 73.18 |
| Fig 4–5 | Số nút/đồ thị con CSQA: đỉnh ~10–15, đa số ≤30, đuôi tới ~80; số cạnh: đỉnh ~10–20, đuôi tới ~160 |
| Fig 9 | Ảnh 50×50: CSQA ≈77–78% (**ước lượng từ biểu đồ**, bài không ghi số) — vẫn hơn base ~7 điểm |

### A3. Điểm không khớp bài ↔ mã (GraphVis-code)
- `scripts/finetune_lora.sh:38` lr **2e-5** (bài: 1e-7); `:31` batch 16/GPU × `:17` 2 GPU = 32 (bài: 4).
- `scripts/finetune_lora.sh:24` `--vision_tower openai/clip-vit-large-patch14` (224 px), LLaVA-1.6 dùng bản 336 px.
- Script không truyền `--image_aspect_ratio` → mặc định `square` (`llava/train/train.py:79`) = cắt giữa, không phải `pad`.
- `llava/train/train.py:723,726` đọc trường `chosen` (định dạng DPO), còn `generate_data.py` ghi `conversations`.
- `llava/train/train.py:830` nạp bằng `LlavaLlamaForCausalLM` dù checkpoint là Mistral.
- Mask loss: token prompt gán `IGNORE_INDEX` (`train.py:283,286` …) → loss chỉ trên token trả lời.
- LoRA target: `find_all_linear_names` (`train.py:172`) bỏ qua `mm_projector`, `vision_tower`, `lm_head`.
- Thiếu: script tách 2 giai đoạn, mã vẽ Graphviz (`import graphviz` bị comment), mã sinh D_v, mã đánh giá 6 tác vụ, mã KAPING.
- `generate_data.py:54` thiếu dấu phẩy → 2 mẫu câu "bậc cao nhất" bị nối, còn 4 mẫu.
- `generate_data.py:60` `\"{node\"}` → không thay tên nút được.
- `generate_data.py:150` bậc đếm theo mọi link; `:198` số cạnh = số bộ ba **không trùng** có quan hệ hợp lệ → có thể không nhất quán.
- `generate_data.py:134` gộp đồ thị theo khối 4 dòng; đường dẫn mặc định là OBQA (4 lựa chọn). CSQA có 5 lựa chọn → cần sửa.
- Ước tính của nhóm: tham số LoRA ≈ 336M (≈4.6% của 7.24B); projector ≈ 21M.

### A4. Phân tích độ phức tạp ảnh (luận điểm trong mục VI)
- Với `pad`, mọi ảnh → 336×336 → 24×24 = 576 token, **không tăng** theo kích thước đồ thị.
- Chữ ~14 px gốc, ảnh rộng 1000 px → ~4.7 px; 2000 px → ~2.4 px (giả định — cần đo kích thước ảnh Graphviz thật).
- Ở 50×50 nhãn chắc chắn không đọc được nhưng QA vẫn tăng → giả thuyết: phần lớn mức tăng đến từ SFT trên tập QA.
  Bài **không có** đối chứng "tinh chỉnh không ảnh" → thực nghiệm C3.

---

## B. SciMKG

Các phát hiện SciMKG đang duyệt nằm trong [SciMKG/docs/FINDINGS.md](../SciMKG/docs/FINDINGS.md).

## C. GraphVis — số liệu dữ liệu đã đo trên máy local
- Đồ thị con CSQA dev (300 câu) theo cấu hình mặc định (bậc ≤ 500, 1 nút trung gian/cặp): trung vị 21 nút / 26 cạnh,
  p90 38 / 53, max 68 / 114, 1,7–1,9% rỗng — gần Fig 4–5 của bài. Không tỉa: trung vị 133 nút.
- Kích thước ảnh Graphviz thật (CSQA dev, 1.198 ảnh có đồ thị, 96 dpi, chữ 14 pt ≈ 18,7 px; đo 18/10/2026 bằng `repro/graphvis/scripts/measure_image_sizes.py`): cạnh dài trung vị 1.349 px (p10 667, p90 2.322, max 4.429); 93% ảnh nằm ngang, rộng/cao trung vị ~2:1. Thu về 336 px: chữ trung vị 4,7 px, 38% ảnh < 4 px, 69% < 6 px; theo số nút ≤10: 11,7 px, 11–20: 5,9, 21–40: 3,6, >40: 2,4 px.
