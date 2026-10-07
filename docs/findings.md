# Những gì đã kiểm chứng (bài báo ↔ mã nguồn)

Dùng file này thay vì đọc lại bài báo. Số dòng tính theo bản mã hiện có trong repo
(`GraphVis-code/` ở commit đầu, `SciMKG/` ở commit `81f2ce2`).

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

### B1. Thiết lập & số liệu (bài báo)
- LLM: GPT-4o, Gemini-1.5-Flash, DeepSeek-V3 (API, 01–04/2025). MLLM ảnh: GPT-4o, Gemini-1.5-Pro.
- Siêu tham số (Table 6): temperature 0, max_length 1000, top_p 0.1, frequency_penalty 0.5, seed 42, iter 5, α 0.7.
- Tập đánh giá: 100 bài học, 4.479 khái niệm. Môi trường ghi: Ubuntu 22.04, RTX 3090, Python 3.7, PyTorch 2.4.0+cu124 (không tương thích).
- Table 4 (P/R/F1/ACC/MCC/AUC): Decomposed-QA .935/.570/.712/.977/**.741**/.885; GPT-NER .530/.931/.681/.923/.621/.938;
  LinkNER .855/.644/.734/.981/.732/.912; UniversalNER .873/.617/.718/.974/.722/.929; Self-Improving .911/.532/.670/.972/.725/.935;
  Q4B+G4B+P3.8B .817/.580/.678/.966/.665/.891; Q8B+G12B+P7B .861/.614/.717/.974/.719/.914;
  Q14B+G27B+P"27B" .879/.637/.739/.982/.730/.926; w/o MLs .769/.342/.477/.963/.509/.845;
  w/o SF-FD .374/.589/.459/.883/.428/.810; w/o cả hai .340/.290/.312/.887/.315/.762; **Full .947/.702/.803/.986/.729/.985**.
- Table 5: lưới iter{1,3,5} × α{0.5,0.7,0.9}; tốt nhất iter=5, α=0.7 (F1 .803).
- Fig 5 lỗi: IO 34%, FE 26%, CE 11%, NR 10%, CO 9% (tổng 90%).
- Table 7: CLIP .83, BLIP .76, X-VLM .81, VQAScore .75, Human .73.
- Thống kê: 1.356 điểm KT, 34.630 khái niệm, 403.400 bộ ba, 763 bài tập, 10.527 ảnh, 10.425 video, 34.630 audio;
  phủ: ảnh 39%, video 80%, audio 100%. Phân bố quan hệ (Fig 3): related to 17.9%, hasImage/isImageOf 17.1% mỗi chiều,
  explanation 8.9%×2, audio 8.9%×2, video 6.0%×2, exercise 0.3%×2.

### B2. Cơ chế thực tế theo mã nguồn
- Tiền xử lý (`Dataprocess/Dataprocess.py`): MP4→MP3 (moviepy) → ASR thủ công (công cụ ngoài, định dạng "发言人 <time>")
  → parse thành `"mm:ss-mm:ss": "nội dung"` → Gemini gộp đoạn theo ngữ nghĩa (giữ nguyên văn) → cắt clip theo mốc.
  Ảnh slide: `Dataprocess/ImageExtractFromPdf.py` gọi dịch vụ tách ảnh PDF (kiểu Stirling-PDF `/api/v1/misc/extract-images`).
- EVIA (`ExtractMultimodalConcepts/NER.py`):
  - Extraction: `:84` chỉ DeepSeek-V3; output tiếng Trung `-xxx`; tập `allEntity` toàn cục → khái niệm gắn với **đoạn đầu tiên** xuất hiện.
  - Validation: mỗi model trong `["gpt-4o","deepseek-v3","gemini-1.5-flash"]` × k=5 vòng: prompt FEEDBACK (từ mơ hồ liên ngành) +
    prompt REFINE (loại ngoài môn, loại dụng cụ thí nghiệm). Mỗi lần sống sót → sc_score += 1 (tối đa 1+15).
    `:202` danh sách đưa vào REFINE luôn là danh sách gốc (không cập nhật giữa các vòng). `:255` `sc_max[index] +1` không gán → no-op.
  - Integration: `:322` `SC_ScoreG = (1 / k * k) * sc_score` = sc_score (không chuẩn hóa); sc_score khởi tạo 1 ⇒ `>= 0.7` luôn đúng ⇒ **không lọc gì**.
  - Augmentation (`ConceptNet.py`): `/c/zh/<concept>` → nút tiếng Anh → 100 cạnh surfaceText → regex (bộ luật **để trống**) → DeepSeek-V3 lọc K12.
  - Giải thích (`Explaination.py`, `Wikipediaapi_model.py`): Baidu Translate zh→en → Wikipedia EN summary → LLM viết ≤20 từ tiếng Trung.
    `Wikipediaapi_model.py:53` trả về dict nhưng được so với chuỗi `"not found"` → nhánh "không tìm thấy" không bao giờ chạy.
- Căn chỉnh ảnh (`MultimodalAlignment/ImageAligment.py:121-125`): GPT-4o-2024-11-20 gọi 3 lần, lấy hợp; prompt 4 tầng
  C4 (khái niệm của bài) > C3 (tổng thể) > C2 (vùng) > C1 (OCR); khái niệm mới từ ảnh được thêm vào. **Không** có MLLM thứ 2 / ngưỡng 0.8.
- Video: khái niệm ↔ clip của đoạn sinh ra nó (mốc thời gian), không có mô hình thị giác.
- Âm thanh (`SpeechGenerator.py:25`): OpenAI `tts-1`.
- Bộ ba (`SciMKGConstruction/GeneratorTriples.py`): cặp thuận–nghịch cho explanation, image, video, audio; `related to` 2 chiều
  khái niệm↔điểm KT; điểm KT↔bài tập, điểm KT↔điểm KT. **Không có cạnh khái niệm–khái niệm.**
- RDF (`RDF/rdf.py`): rdflib, namespace `edukg`, 7 lớp; `:65,80` khóa `"has an video"` ≠ `"has a video"` khi sinh bộ ba.

### B3. Điểm phê bình số liệu
- "Vượt mọi baseline" nhưng MCC .729 < Decomposed-QA .741, LinkNER .732.
- ACC .88–.99 cho mọi cấu hình → mất cân bằng lớp; AUC tính từ điểm gì không rõ.
- Văn bản: F1 .741, ΔF1 .062; bảng: .739, .064. Chú thích "P-27B = Phi-3-medium" (thực tế ~14B).
- CLIP score .83 cao bất thường nếu là cosine thô. Không có quy trình gán nhãn vàng, không có độ đồng thuận chuyên gia.

### B4. Kết quả đã chạy trên máy local (không cần API/GPU)
- **Dữ liệu Zenodo 19691725:** `Groundtruth.zip` khớp tên với 98 thư mục bài trong `RawData.zip`: 1.526 đoạn,
  3.896 khái niệm vàng (bài báo ghi 100 bài, 4.479 khái niệm). 64 bài có file phiếu `_Validationner` của tác giả
  (điểm 1..16 = 1 + 3 LLM × 5 vòng — khớp mã nguồn). Tên ảnh PNG đánh số toàn cục theo môn, khớp `SciMKG.zip`.
- **Chấm lại đầu ra LLM gốc của tác giả** (`smkg.cli score-authors`, điểm chuẩn hóa (s−1)/15, vũ trụ = nhãn vàng ∪ ứng viên):

| α | P | R | F1 | ACC | MCC | AUC |
|---|---|---|---|---|---|---|
| 0,5 | .891 | .959 | .924 | .911 | .819 | .949 |
| 0,7 | .953 | .740 | .833 | .831 | .690 | .949 |
| 0,9 | .963 | .543 | .695 | .728 | .547 | .949 |

  So với bài (α=0,7): P .947, R .702, F1 .803, MCC .729 — rất sát; nhưng ACC .831 so với .986 của bài →
  bài gần như chắc chắn tính ACC trên một tập mẫu âm lớn hơn nhiều (vd mọi từ/cụm từ). Ở α=0 recall .995:
  nhãn vàng gần như nằm trọn trong tập ứng viên trích xuất → nhãn vàng được tuyển từ ứng viên. α=0,5 cho F1 cao hơn α=0,7.

## C. GraphVis — số liệu dữ liệu đã đo trên máy local
- Đồ thị con CSQA dev (300 câu) theo cấu hình mặc định (bậc ≤ 500, 1 nút trung gian/cặp): trung vị 21 nút / 26 cạnh,
  p90 38 / 53, max 68 / 114, 1,7–1,9% rỗng — gần Fig 4–5 của bài. Không tỉa: trung vị 133 nút.
- Ảnh Graphviz mẫu rộng ~2.000 px → khi đưa về 336 px chữ còn ~2,5 px (minh chứng cho mục VI).
