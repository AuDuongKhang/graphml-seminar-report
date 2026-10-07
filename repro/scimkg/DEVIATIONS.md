# Khác biệt so với bài báo SciMKG (nguồn cho mục XII-A của báo cáo)

| # | Bài báo / mã tác giả | Nhóm làm | Lý do |
|---|---|---|---|
| 1 | GPT-4o, Gemini-1.5-Flash, DeepSeek-V3 (API, 01–04/2025) | Mặc định Qwen3-14B + Gemma3-27B + Phi-3-medium (bộ mô hình mở lớn nhất trong Bảng 4 của bài) qua Ollama, lượng tử q4. Bộ của bài vẫn chạy được nếu có key (`USE_PAPER_MODELS`); Gemini-1.5 có bản thay thế khai báo trong `models.yaml` | Nhóm chỉ có tài nguyên miễn phí, chưa có API key |
| 2 | Bước tích hợp: `SC_ScoreG = (1/k*k) * sc_score` ≥ 0,7 (luôn đúng → không lọc) | Điểm = số (LLM, vòng) giữ khái niệm / (\|M\|·K), giữ nếu ≥ α | Đúng ý "tần suất" của bài; mã gốc có lỗi thứ tự phép tính |
| 3 | Mã gốc: chỉ DeepSeek-V3 trích xuất | Cả 3 LLM trích xuất (theo bài); `--extractors` để chạy như mã gốc | Theo mô tả trong bài |
| 4 | Ablation chạy riêng | w/o MLs, w/o SF-FD, w/o cả hai và lưới K × α được tính lại từ log phiếu của lần chạy đầy đủ. w/o MLs = phiếu của LLM đầu trên các ứng viên do chính nó trích (prompt sửa đã thấy pool hợp) | Không tốn thêm lời gọi; sai khác nhỏ ghi rõ |
| 5 | Tập đánh giá 100 bài, 4.479 khái niệm; quy trình gán nhãn không công bố | `Groundtruth.zip` trên Zenodo, khớp tên với RawData: 98 bài, 1.526 đoạn, 3.896 khái niệm. Mặc định 30 bài chia đều 3 môn (seed 42) | Bản công bố khác số liệu trong bài; quota GPU |
| 6 | P/R/F1, ACC, MCC, AUC không định nghĩa | P/R/F1 micro trên tập khái niệm; ACC/MCC/AUC trên vũ trụ = nhãn vàng ∪ pool ứng viên ∪ dự đoán; AUC theo điểm tỉ lệ phiếu (baseline nhị phân → balanced accuracy); so khớp sau chuẩn hóa NFKC | Ghi rõ trong báo cáo |
| 7 | Phân loại lỗi FE/IO/CE/CO/NR không định nghĩa | FE: không tách được danh sách; NR: rỗng/quá giờ/lỗi API; IO: khái niệm không có trong văn bản; CE: có trong văn bản nhưng không thuộc nhãn vàng; CO: khái niệm vàng bị bỏ sót | Bài không mô tả |
| 8 | Baseline LinkNER | Không chạy | Cần mô hình NER cục bộ đã huấn luyện cho miền này |
| 9 | Baseline Self-Improving: truy hồi ví dụ bằng embedding | Cosine trên bigram ký tự | Văn bản tiếng Trung; không cần thêm mô hình |
| 10 | Phi-3-small (7B) trong bộ cỡ vừa | Mặc định thay bằng Phi-3-mini; có ô tùy chọn chạy Phi-3-small bằng vLLM | Không có trên Ollama |
| 11 | Căn chỉnh ảnh: MLLM thứ hai chấm độ tin cậy > 0,8 (bài) / GPT-4o gọi 3 lần lấy hợp (mã) | `regen-align` theo mã gốc (3 lần, lấy hợp) với mô hình đa phương thức mở | Mã không có bước chấm tin cậy |
| 12 | Table 7: CLIP, BLIP-ITM, X-VLM-ITM, VQAScore, 5 chuyên gia | CLIPScore (cosine thô và 2,5·max(cos,0)), BLIP-ITM; VQAScore tùy chọn; X-VLM không chạy (không có bản HF); chấm người = 2 thành viên, 100 cặp, có Cohen κ | Khả năng tính toán; bài không nêu cách chuẩn hóa CLIP score |
| 13 | Khái niệm tiếng Trung đưa thẳng vào CLIP/BLIP? (không nêu) | Dịch sang tiếng Anh bằng LLM (có cache) trước khi chấm | CLIP/BLIP chỉ hiểu tiếng Anh |
| 14 | Mô hình Qwen3 sinh khối `<think>` | Bỏ khối `<think>` trước khi tách danh sách | Tránh lỗi định dạng giả |
