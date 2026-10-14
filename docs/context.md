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

Cập nhật cấu trúc 12/10/2026: bỏ tóm tắt và từ khóa chung; sau tiêu đề và tác giả, vào thẳng **Phần A — GraphVis**, rồi **Phần B — SciMKG** ở trang mới. Mỗi phần có giới thiệu, kiến thức nền và kết luận riêng; không dùng ba mục này ở cấp báo cáo chung. Mỗi phần đánh số mục lại từ I: Phần A I–IX, Phần B I–V (Giới thiệu, Công trình liên quan, Phương pháp, Thực nghiệm, Kết luận). Phân công, tài liệu tham khảo và phụ lục dùng chung ở cuối báo cáo.

## 5. Quyết định cho lần chạy thử SciMKG của Nhi (10/10/2026)

- Mục tiêu: kiểm tra tính đúng đắn của mã theo khung xây dựng SciMKG và thăm dò khả thi trên một bài tiếng Việt; chưa phải tái lập toàn bộ kết quả bài báo.
- Không dùng mã PR #1 cho lần chạy thử. Đọc bài và đối chiếu mã gốc tại commit `81f2ce230c2aac2f433e2c86a01ac221c7567405`; triển khai độc lập ở `papers/SciMKG/pilot/`.
- Kiểm tra **một mẫu nhỏ của tác giả trước**, ưu tiên `Example.zip`; không chạy toàn bộ tập. Chỉ sau đó thử bài tiếng Việt.
- Dùng Codex CLI với đăng nhập thuê bao ChatGPT để thăm dò; không tự chuyển sang API trả phí. Một mô hình và TTS cục bộ là các điều chỉnh phải ghi rõ; token/thời gian không đồng nghĩa với chi phí API thực đo.
- Toàn bộ tệp pilot nằm trong `papers/SciMKG/pilot/`. Báo cáo hợp nhất tiếng Việt: `papers/SciMKG/pilot/report/pilot-report.html` (kế hoạch, duyệt đầu ra, kết quả, đối chiếu mã, nguồn bằng chứng). Kết quả giả lập không thay cho đánh giá bài học; chấm nội dung và W&B vẫn cần hoàn tất.

Cập nhật 11/10/2026: người dùng giao thực hiện các mục và sẽ duyệt cuối. Đã chạy trên hai trích đoạn nhỏ đúng thứ tự; chưa chạy toàn bài. Nhãn phạm vi tác giả và nhãn Việt đều cần người duyệt, số chấm hiện là tạm. Clip Việt 49 giây đã lấy riêng và kiểm tra tệp/mốc thời gian; cần người duyệt nội dung, video tác giả vẫn thiếu. Hình Việt từ lý thuyết cùng bài, không phải ảnh chụp video. ConceptNet API lỗi nên dùng cạnh trên trang tra cứu và dịch nhãn có ghi nguồn. W&B mới ngoại tuyến vì chưa có cấu hình tài khoản/dự án công khai; phần duyệt cục bộ ở `papers/SciMKG/pilot/report/pilot-report.html#review`.

Cập nhật phần duyệt 11/10/2026: Nhi không biết tiếng Trung. Mẫu tác giả có bản dịch tiếng Việt để đọc và xem liên kết ảnh; nội dung/ngữ âm gốc chưa được người biết tiếng Trung duyệt. Bản dịch chỉ là lớp trình bày, không sửa nhãn, dự đoán hoặc số chấm. Phần bài Việt vẫn duyệt nội dung tiếng Việt gốc.

Đối chiếu đầu ra công bố 11/10/2026: Nhi chưa có kiến thức chuyên ngành để tự chấm mẫu sinh học. Báo cáo ghi riêng bằng chứng kỹ thuật và chất lượng chưa chấm, không yêu cầu người dùng xác nhận mọi nội dung. Khớp tên 5/6 nhãn tạm: “hoạt động sống” chưa khớp với “hoạt động sống của tế bào”; 12 dự đoán ngoài nhãn tạm chưa được phân xử. Ba tệp trung gian thuộc đúng bài đã có; tính lại phiếu giữ 3 khái niệm trong hai đoạn, còn tệp tích hợp công bố ghi 1, chưa rõ cấu hình sinh. Ví dụ ti thể có ảnh/giải thích/MP3/mốc video nhưng chưa xác nhận cùng trích đoạn; không dùng làm đáp án đầy đủ. Bằng chứng tại `pilot/report/pilot-report.html#author-output-comparison` và `pilot/results/author-output-comparison.json` dưới `papers/SciMKG/`.

Phạm vi tiền xử lý, chốt 11/10/2026: pilot tiếp tục dùng văn bản/ảnh đã chuẩn bị; không bổ sung mã xử lý video/PDF gốc lúc này. Trước lượt dữ liệu thực sau pilot, mã chạy phải có lệnh tách âm thanh, nhận hoặc tạo bản chép lời có thời gian, phân đoạn, lấy ảnh PDF và cắt clip, rồi kiểm tra đầu vào. Chưa coi việc kiểm tra đầu vào của pilot là đã kiểm chứng toàn bộ chuỗi tiền xử lý; yêu cầu này được ghi ngay trong bước 01 của báo cáo hợp nhất.
