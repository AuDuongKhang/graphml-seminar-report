# Ngữ cảnh dự án (tóm tắt từ các file nguồn không đưa lên git)

## 1. Yêu cầu báo cáo của môn học (tóm tắt `requirement_report.txt`)

- Nộp **1 file PDF** tên `MSHV1-MSHV2.pdf` → `25C15045-25C15055.pdf`; **< 20 MB**; **tối đa 25 trang**
  tính cả trang bìa, tài liệu tham khảo và phụ lục. Template IEEE (LaTeX).
- Báo cáo gồm **cả hai bài** (bài đã seminar + bài "chỉ báo cáo"), trình bày như **hai phần của một báo cáo**,
  không phải hai báo cáo riêng. Giới hạn trang áp cho tổng.
- **Không sao chép** nhiều câu chữ của bài gốc hay nguồn công khai; phải diễn đạt cách hiểu của nhóm.
  Được trực quan hóa lại khái niệm, nhưng cân đối kích thước, đảm bảo chất lượng/độ phân giải hình.
- **Tái lập thực nghiệm** phải được ghi nhận bằng **Weights & Biases**. Trong từng phần thực nghiệm của báo cáo,
  gắn link tới kết quả tương ứng (kết quả chính → link thực nghiệm chính; ablation → link ablation; ...).
  Link phải **xem công khai được**. Project W&B chế độ Teams: dùng cross-project report
  (https://docs.coreweave.com/products/wandb/reports/cross-project-reports).
- Chỉ dùng **một ngôn ngữ** (đã chọn: tiếng Việt), thống nhất, không pha tạp.
- Mã nguồn tái lập quản trị trên **GitHub**, dẫn link trong báo cáo ở phần tương ứng.
  **README.md** phải nói rõ cách chạy lại: môi trường, phần cứng, câu lệnh, file Python/Notebook nào...
  **Không** đưa kết quả thực nghiệm hay mô tả giải pháp vào README (những thứ đó thuộc báo cáo).

## 2. Nhận xét của giảng viên sau buổi seminar GraphVis

Slide thuyết trình (Canva) có 18 trang. Hai nhận xét:

| Slide | Nội dung slide | Nhận xét | Đã đưa vào báo cáo ở |
|---|---|---|---|
| 3 | "Vì sao là đồ thị, không phải văn bản?" — ConceptNet 799.273 nút, 2.487.810 cạnh, không nhét cả đồ thị vào mô hình được nên phải cắt đồ thị con | "Mức độ hình ảnh là bao nhiêu thì mô hình trả lời sai? (thông số kích thước, ...)" | Mục VI `sec:gv-complexity` (+ thực nghiệm C1–C3 trong `docs/experiments.md`) |
| 11 | Sơ đồ toàn cảnh: dựng dữ liệu ảnh (D_g, D_v) → giai đoạn 1 học đọc hình → mang tham số → giai đoạn 2 học trả lời | "Cách huấn luyện mô hình ở slide này?" | Mục IV-E `sec:gv-training` (hàm mục tiêu, khối đóng băng/LoRA/projector, lịch, siêu tham số, Hình kiến trúc) |

Quy tắc: lồng vào như thông tin chi tiết, **không** nhắc "thầy hỏi"/"slide" trong báo cáo.

## 3. Hai bài báo

- **GraphVis**: Y. Deng, C. Ye, Z. Huang, M. D. Ma, Y. Kou, W. Wang. *GraphVis: Boosting LLMs with Visual
  Knowledge Graph Integration*. NeurIPS 2024.
  PDF: https://proceedings.neurips.cc/paper_files/paper/2024/file/7cb04f510593c9ba30da398f5e0a7e7b-Paper-Conference.pdf
  Mã: thư mục `GraphVis-code/` (bản tải về, Apache-2.0).
- **SciMKG**: T. Lu, Z. Wang, Y. Zhou, Y. Guan, Z. Bai, J. Du. *SciMKG: A Multimodal Knowledge Graph for Science
  Education with Text, Image, Video and Audio*. AAAI 2026, vol. 40(18), pp. 15466–15474,
  DOI 10.1609/aaai.v40i18.38574. Mã: https://github.com/kg-bnu/SciMKG (submodule `SciMKG/`).
  Dữ liệu: Zenodo (https://zenodo.org/records/17578391, README cũng nhắc records/19691725). Demo: https://demo.scimkg.lutong.space/

## 4. Quyết định đã chốt (07/10/2026)

- Ngôn ngữ: tiếng Việt; LaTeX biên dịch bằng XeLaTeX.
- Tiêu đề: "Báo cáo hai mô hình: GraphVis và SciMKG". Bìa: logo ĐH KHTN, tháng 10 năm 2026.
- GraphVis: **tái lập tất cả thí nghiệm** trên **Google Colab**, log W&B công khai.
  Máy local của Khang chỉ có RTX 3060 Laptop 6 GB — không đủ fine-tune LLaVA-1.6-7B.
- SciMKG: tái lập **đúng cấu hình bài** (GPT-4o, Gemini-1.5-Flash, DeepSeek-V3; MLLM GPT-4o/Gemini-1.5-Pro)
  nếu còn dùng được; nếu không, ghi rõ trong báo cáo và dùng LLM mở (Qwen3/Gemma3/Phi-3 như Bảng 4 của bài).
- Hai bài chọn ngẫu nhiên → không viết phần liên kết/khung chung.
