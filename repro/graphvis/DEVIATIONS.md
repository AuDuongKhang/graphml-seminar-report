# Khác biệt so với bài báo GraphVis (nguồn cho mục VII-A của báo cáo)

| # | Bài báo / mã tác giả | Nhóm làm | Lý do |
|---|---|---|---|
| 1 | Fork LLaVA cũ (deepspeed, flash-attn); `train.py` đọc trường `chosen`, nạp Mistral bằng lớp Llama | Vòng huấn luyện viết lại bằng HF `transformers` + `peft` trên `llava-hf/llava-v1.6-mistral-7b-hf` | Mã tác giả không chạy được nguyên trạng (`docs/findings.md` §A3) |
| 2 | LoRA bf16 trên 4×A6000 | Mặc định QLoRA 4-bit NF4 + fp16 (T4/P100 không có bf16); tham số học giữ fp32 | Chỉ có GPU miễn phí. Có thể chạy đúng LoRA bf16 bằng `--set quant=none dtype=bf16` |
| 3 | `image_aspect_ratio=pad`: 1 ô 336 px → 576 token | Đệm vuông bằng màu trung bình CLIP + `image_grid_pinpoints=[[336,336]]` → ảnh gốc 336 + 1 ô 336 (1.152 token, cùng nội dung) | HF LlavaNext luôn ghép ảnh gốc với các ô AnyRes |
| 4 | Tốc độ học 1e-7 (bài) / 2e-5 (script); batch 4 (bài) / 32 (script) | lr 2e-5, batch toàn cục 4 (1 × tích lũy 4) | 1e-7 trong 1 epoch gần như không cập nhật; batch theo bài |
| 5 | Đồ thị con từ file `graph.jsonl` của MHGRN (không công bố), tỉa đường đi bằng điểm embedding | Tự truy hồi từ `grounded.jsonl` + `conceptnet.en.csv` của QA-GNN: đường đi ≤ 2 bước giữa khái niệm câu hỏi và đáp án; bỏ nút có bậc > 500; mỗi cặp (q, a) giữ 1 nút trung gian bậc thấp nhất; tối đa 80 nút | Phân bố trên CSQA dev: trung vị ~21 nút / ~26 cạnh, p90 ~38 / ~53, max ~68 / ~114 — gần Fig 4–5 của bài. Không tỉa: trung vị 133 nút |
| 6 | ConceptNet dạng networkx pickle | Đồ thị đã tỉa dựng lại từ CSV theo đúng luật của QA-GNN (blacklist, bỏ `hascontext`, bỏ vòng, bỏ trùng), lưu dạng mảng CSR | Pickle cần > 4 GB RAM |
| 7 | Mẫu câu D_g: `generate_data.py` thiếu dấu phẩy (dòng 54), `{node\"}` (dòng 60); bậc đếm theo link, số cạnh theo bộ ba không trùng | Mẫu câu đúng như Phụ lục A; bậc và số cạnh đều tính trên cùng tập bộ ba vẽ trong ảnh | Sửa lỗi |
| 8 | Split không nêu | CSQA: train chính thức (9.741) / đánh giá trên dev chính thức (1.221). OBQA: train 4.957 / test 500 | CSQA test không có nhãn |
| 9 | `lite`: D_g 600 ảnh × 6, D_v 2.000 câu | Bài dùng toàn bộ tập train | Quota GPU miễn phí; `--scale full` để chạy như bài |
| 10 | Đánh giá VQA bằng script LLaVA; MMBench chấm trên server | Chấm local: ScienceQA-IMG test; MMBench-EN dev (CircularEval + single-pass); POPE 3 thiết lập. `lite`: MMBench 400 câu gốc, POPE 500 câu/thiết lập | Không có server |
| 11 | "Base LVLM" không rõ có ảnh | Báo cả `text` và `image` | Minh bạch |
| 12 | KAPING tự cài lại (không công bố) | MPNet (`all-mpnet-base-v2`), k = 10, chèn trước "Question:" | Theo bài KAPING |
| 13 | Đọc ảnh: "exact match", trung bình theo mẫu cho liệt kê | Số/bậc: khớp số nguyên; nút bậc cao nhất: tên thuộc tập bậc max **và** đúng bậc; liệt kê nút/bộ ba: tỉ lệ phần tử đúng | Bài không mô tả chi tiết |
| 14 | Ảnh 50×50 | Đệm vuông rồi thu về 50×50 (LANCZOS) cả khi train lẫn eval | Bài không mô tả chi tiết |
| 15 | 4×A6000 (song song dữ liệu) | Huấn luyện trên **1 GPU** kể cả khi Kaggle có 2×T4 (bnb 4-bit + PEFT không chạy được dưới `nn.DataParallel`); batch toàn cục vẫn là 4 | Ổn định |
| 16 | Adapter lưu fp32 | Adapter `final/` lưu fp16 (~0,7 GB), checkpoint trung gian bị xóa sau khi xong giai đoạn | Giới hạn đĩa Kaggle 20 GB / Drive 15 GB |
| 17 | Bảng 4: đọc ảnh trên ảnh test CSQA | `lite`: 600 câu (~100 ảnh × 6 tác vụ), sinh tối đa 384 token — danh sách bộ ba rất dài (đồ thị > ~40 cạnh) có thể bị cắt, làm điểm liệt kê thấp hơn thực tế | Thời gian GPU |
| 18 | Thực nghiệm độ phân giải (thiết kế của nhóm, không có trong bài) | `lite`: mỗi điểm (chế độ ảnh × kích thước) đánh giá trên 300 câu đầu của CSQA dev thay vì đủ 1.221 câu | Vừa một lượt chạy Kaggle ≤ 12 giờ |
