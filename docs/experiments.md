# Danh sách thực nghiệm tái lập

Mỗi dòng = một W&B run/report công khai, gắn vào đúng mục của báo cáo bằng `\wandb{url}`.
Cập nhật cột *Trạng thái* và *W&B* ngay khi xong. Không ghi số liệu vào README (chỉ vào báo cáo).

Cấu trúc báo cáo cập nhật 12/10/2026: mỗi phần có giới thiệu, kiến thức nền và kết luận riêng; bỏ tóm tắt và từ khóa chung, vào thẳng Phần A sau tiêu đề/tác giả. Mục tái lập GraphVis vẫn là VII (`sec:gv-repro`); mục tái lập SciMKG chuyển từ XII sang VI (mỗi phần đánh số lại từ I) (`sec:sm-repro`). Thay đổi này chỉ cập nhật vị trí trình bày, không thay trạng thái hay kết quả thực nghiệm.

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
| S0 | Chuẩn bị tập đánh giá + nhãn vàng (bài không công bố → tự gán trên tập con, mô tả quy trình) | Zenodo | VI-A | code sẵn sàng | – |
| S1 | Table 4: EVIA đầy đủ + baseline (Decomposed-QA, GPT-NER, Self-Improving, UniversalNER) + chấm lại đầu ra gốc của tác giả (`score-authors`, không tốn API) | GPT-4o/Gemini-1.5-Flash/DeepSeek-V3 nếu còn API | VI-B | code sẵn sàng | – |
| S2 | Table 4: w/o MLs, w/o SF-FD, w/o cả hai | | VI-B | code sẵn sàng | – |
| S3 | Table 4: bộ LLM mở Qwen3/Gemma3/Phi-3 theo kích thước | Colab/Ollama | VI-B | code sẵn sàng | – |
| S4 | Table 5: lưới iter{1,3,5} × α{0.5,0.7,0.9} | | VI-C | code sẵn sàng | – |
| S5 | Fig 5 phân loại lỗi; Table 7 CLIP/BLIP/X-VLM/VQAScore cho cặp khái niệm–ảnh | | VI-D | code sẵn sàng | – |

Đã xử lý trong code: tích hợp chuẩn hóa theo số (LLM, vòng) chạy thành công; cả 3 LLM trích xuất (`--extractors`
để chạy như mã gốc); `regen-align` theo mã gốc (không có bước chấm tin cậy 0,8) — ghi rõ trong VI-A theo
`repro/scimkg/DEVIATIONS.md`. Trước khi chạy: `check-models` để biết mô hình nào dùng được; mô hình có `fallback`
được tự thay và tên mô hình thực tế được ghi trong W&B (`resolved_models`).

## SciMKG — Ex01/Ex02 (cập nhật 13/10/2026)

Phạm vi hiện hành ở [CHECKLIST của fork](../SciMKG/CHECKLIST.md); các mục S0–S5 phía trên là kế hoạch tái lập cũ. Ex01/Ex02 dùng nhãn công bố của tác giả đã cố định theo bài.

| ID | Công việc | Trạng thái | W&B |
|---|---|---|---|
| Chuẩn bị chung | Sáu bài, mỗi môn hai bài; văn bản và nhãn tác giả tách riêng | **Xong:** 75 đoạn, 188 nhãn theo bài; CRC/SHA-256 đã đối chiếu, 24 kiểm thử CPU đạt, chạy lại giữ nguyên 19 hash | Chưa chạy mô hình |
| Ex01 | Chi phí toàn bài `nervous-system` | **Xong, dùng trong báo cáo:** lượt `gpt-6-luna` chạy đủ mọi bước (14 đoạn, 27 ảnh): 335 lượt gọi, không có lần thử lại; 4.325.932 token vào, trong đó 2.779.392 đã lưu sẵn; 14.909 token ra; khoảng 32 phút. Một bài khoảng 0,19–0,23 USD theo giá tra ngày 14/10/2026. Video/PDF gốc và tiền xử lý lịch sử còn thiếu; chất lượng chưa chấm. Lượt 13/10 đã xóa khỏi máy. | Chưa kiểm tra công khai: [run gpt-6-luna](https://wandb.ai/nhinbm-hcmus/SciMKG/runs/7b20992d11ea) |
| Ex02 | Sáu bài × ba cấu hình: đầy đủ, tắt verification, tắt augmentation | Chờ chạy 18 lượt; chấm sau Integration, trước Augmentation; báo cáo từng bài và trung bình cộng P/R/F1 theo bài | Chưa có |

Dữ liệu: [manifest](../SciMKG/data/MOOC/manifest.json). Nguồn, danh sách và giới hạn: [MOOC](../SciMKG/docs/MOOC.md). Lệnh chuẩn bị/kiểm tra: `python3 scripts/preprocessing/prepare_mooc.py` / `--check` trong fork. Hai bài Vật lý dự kiến ban đầu bị cắt dở trong nguồn nên đã thay trước khi chạy mô hình bằng áp suất khí quyển và lực đẩy. Bộ nhỏ này chưa đại diện toàn tập; chất lượng Augmentation chưa có nhãn tương ứng.

Báo cáo Ex01 trên mục **Reports**: [Token và chi phí một bài học](https://wandb.ai/nhinbm-hcmus/SciMKG/reports/Ex01-Token--VmlldzoxODEwMzc1NA==), công khai đã kiểm tra không đăng nhập. Mở đầu giải thích đầu vào, kết quả và giá API giả định; bảng kỹ thuật được thu gọn. Có tổng 5.837.546 token vào, 39.332 token ra, bảng từng bước và biểu đồ hai nhánh; 1.878.144 token cache dịch vụ đã nằm trong đầu vào. Báo cáo giữ riêng chi phí phát sinh và dựng riêng giả định, ghi rõ 9 lần thiếu usage. Tạo từ nhật ký cũ, không thêm suy luận.

Rà soát mã Ex01 sau chạy ngày 13/10/2026: replay 427 cache lịch sử,
không thêm suy luận hay đổi hash journal/graph. **17/40 nhãn bổ sung** trùng E/V/I sau chuyển giản thể/phồn thể;
40 là số nhãn thô, chưa phải 40 khái niệm mới về nghĩa. Mã protocol 2 sửa chuẩn hóa/bí danh, dừng khi hết hạn mức,
cache, kiểm tra đầu vào và tính chi phí; có trạng thái riêng từng bước. Mã và prompt kỹ thuật dùng tiếng Anh.
Đã rà argument và bỏ phần dư; [chi tiết review](../SciMKG/docs/EX01.md#rà-soát-mã-sau-ex01).
Lượt `gpt-6-luna` protocol 3 đã chạy đủ (14/10/2026) và là số liệu của báo cáo; run W&B chưa kiểm tra quyền công khai.
Bộ test Ex01 và dependency `pytest` đã gỡ theo yêu cầu người dùng sau review; mã đã sửa được giữ lại.
Script audit riêng cũng đã gỡ. Dữ liệu chuẩn bị sinh ra ở `SciMKG/runs/preprocessing/`, lượt chạy ở `SciMKG/runs/pipeline/<mô hình>/<bài>/`, lượt 13/10 đã xóa khỏi máy theo quyết định ngày 14/10;
toàn bộ `runs/` không đưa vào Git. Mã thực nghiệm ở `SciMKG/experiments/ex01/` cần commit.
`scripts/experiments/ex-01/run.py` chỉ khởi chạy CLI; khai báo lệnh nằm trong `experiments/ex01/cli.py`,
thao tác W&B nằm trong `tracking.py`. Đã cập nhật import và dấu phiên bản mã sau chuyển thư mục.
CLI đã rút còn `run` và `report`; chuẩn bị mặc định và kiểm tra cần thiết nằm trong luồng `run`.

## Quy ước W&B

- Entity/project: `<điền>` — đặt project **public** (hoặc Teams + cross-project report công khai).
- Tên run: `<ID>-<mô tả ngắn>`, ví dụ `G4-joint`, `S2-wo-sffd`.
- Log tối thiểu: config đầy đủ (siêu tham số, seed, commit hash), metric cuối, bảng dự đoán mẫu.

## Lần chạy thử SciMKG độc lập của Nhi (10/10/2026)

Quyết định mới cho lần chạy thử này: không dùng mã PR #1, mẫu nhỏ tác giả trước và một bài tiếng Việt sau, Codex dùng thuê bao. Báo cáo hợp nhất: [pilot-report.html](../papers/SciMKG/pilot/report/pilot-report.html). Mã và lệnh chạy: [pilot/README.md](../papers/SciMKG/pilot/README.md).

| ID | Công việc | Trạng thái | W&B |
|---|---|---|---|
| P0 | Đối chiếu bài ↔ mã gốc, triển khai và kiểm thử độc lập | Đối chiếu trực tiếp 11 tệp; **38 kiểm thử CPU đạt**. Dữ liệu kiểm thử giả lập; đối chiếu lượt thật ghi riêng ở P1/P2. | Ngoại tuyến |
| P1 | Kiểm tra quy trình trên một mẫu nhỏ của tác giả | Đã tải mẫu đúng hash; lấy thêm chỉ thành phần một bài qua Range. Chạy hai đoạn 06:35–08:07: 17 khái niệm + 2 bổ sung, 61 nút/132 cạnh, 19 WAV; 46 lời gọi thật. Thiếu video; nhãn phạm vi chưa duyệt. | Ngoại tuyến `vl2j1ruo` |
| P2 | Kiểm tra quy trình trên một bài tiếng Việt | Sau P1, chạy hai đoạn OLM 01:14–02:03: 10 khái niệm + 1 bổ sung, 38 nút/108 cạnh, 11 WAV; 36 lời gọi thật. Clip 49 giây đã kiểm tra tệp/mốc; nhãn Codex gán tạm, nội dung chờ duyệt. | Ngoại tuyến `amwom3iv` |
| P3 | Hai người chấm nội dung, nghe âm thanh, công khai kết quả | **Chờ người duyệt và video đúng bài tác giả**. [Phần ghi nhận xét](../papers/SciMKG/pilot/report/pilot-report.html#review) có ảnh/âm thanh/clip Việt và bản dịch hỗ trợ đọc mẫu tác giả; W&B SDK lưu hai lượt ngoại tuyến, chưa có dự án/link công khai. | Chưa công khai |

Không dùng các kết quả chấm tác giả trước đây làm kết quả mới của lần chạy thử này. Các mục S0–S5 phía trên là kế hoạch tái lập cũ, chưa phải các lượt đã chạy của P0–P3.

Cập nhật 11/10/2026: tổng **82 lời gọi thật**, 1.169.622 token đầu vào, 8.155 token đầu ra, 641,66 giây tổng thời gian CLI; token có ngữ cảnh hệ thống. Dùng xác thực ChatGPT; tiền API chưa đo. Chạy lại cache không thêm suy luận và không đổi hash JSON/RDF. P/R/F1 hiện chỉ là phép chấm trên nhãn tạm, **không phải kết quả chất lượng chính thức**. Chưa có thí nghiệm loại bỏ thành phần độc lập; cả bốn đoạn giữ nguyên ứng viên qua năm vòng. Bằng chứng/hash và các hạn chế ở [báo cáo pilot](../papers/SciMKG/pilot/report/pilot-report.html#results).

Phần duyệt mẫu tác giả dùng bản dịch tiếng Việt do Codex tạo; kiểm chứng bản gốc và phát âm tiếng Trung cần người biết ngôn ngữ. JSON nhận xét ghi rõ căn cứ đọc. Dữ liệu và các số đo P0–P2 giữ nguyên.

Đã bổ sung [bảng đối chiếu đầu ra tác giả](../papers/SciMKG/pilot/report/pilot-report.html#author-output-comparison) và bằng chứng nguồn/mã băm, không phát sinh lời gọi LLM. Khớp 5/6 tên chưa chứng minh mọi dự đoán đúng. Tính lại phiếu của hai đoạn giữ 3 khái niệm, còn tệp tích hợp công bố ghi 1; cấu hình sinh chưa rõ. Mẫu ti thể có tài nguyên tham chiếu, chưa phải nhãn chuẩn đa phương thức của đúng trích đoạn. Không yêu cầu Nhi tự chấm chuyên môn chưa có; phần chưa thể xác minh ghi rõ chưa chấm.

Trước khi chạy dữ liệu thực sau pilot: bổ sung và kiểm chứng lệnh chuẩn bị đầu vào từ video/PDF (âm thanh, bản chép lời có thời gian, phân đoạn, ảnh, clip). Theo người dùng ngày 11/10/2026, việc này nằm sau pilot; hiện giữ nguyên mã và dữ liệu của lượt đã chạy. Xem [điều kiện ở bước 01](../papers/SciMKG/pilot/report/pilot-report.html#real-data-preprocessing).

Đồng bộ EVIA ngày 11/10/2026: pilot gọi trực tiếp mã fork qua `main_evia.py`; chạy `python audit_run.py --check-main-evia` để đối chiếu phản hồi cũ trên mẫu tác giả trước, tiếng Việt sau. Đã kiểm tra cờ nguồn, ngưỡng/lỗi và JSON/RDF; không gọi LLM mới, không thay đầu ra lịch sử. Lượt mới lưu ở `runs/main-evia/`. ConceptNet còn thiếu luật, API Wikipedia trực tiếp chưa kiểm chứng; [mục đồng bộ và bằng chứng E11](../papers/SciMKG/pilot/report/pilot-report.html#main-evia-check).
