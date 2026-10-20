# Danh sách thực nghiệm tái lập

Mỗi dòng = một W&B run/report công khai, gắn vào đúng mục của báo cáo bằng `\wandb{url}`.
Cập nhật cột *Trạng thái* và *W&B* ngay khi xong. Không ghi số liệu vào README (chỉ vào báo cáo).

Cấu trúc báo cáo cập nhật 12/10/2026: mỗi phần có giới thiệu, kiến thức nền và kết luận riêng; bỏ tóm tắt và từ khóa chung, vào thẳng Phần A sau tiêu đề/tác giả. Mục tái lập GraphVis vẫn là VII (`sec:gv-repro`); mục tái lập SciMKG chuyển từ XII sang VI (mỗi phần đánh số lại từ I) (`sec:sm-repro`). Thay đổi này chỉ cập nhật vị trí trình bày, không thay trạng thái hay kết quả thực nghiệm.

Trạng thái: `todo` · `đang chạy` · `xong` · `bỏ (lý do)`

## GraphVis (Kaggle/Colab GPU miễn phí)

Chạy bằng `repro/graphvis/notebooks/GraphVis_Repro.ipynb` — mỗi ID có một ô. Sai khác: `repro/graphvis/DEVIATIONS.md`.

| ID | Thực nghiệm (bài) | Cấu hình | Mục báo cáo | Trạng thái | W&B |
|---|---|---|---|---|---|
| G0 | Dựng dữ liệu: truy hồi 2-hop, vẽ Graphviz, sinh D_g (6 tác vụ) và D_v | CSQA, OBQA | VII-A | **xong** (lượt 1, 15/10/2026, `lite`): CSQA train D_g 3.580 / D_v 2.000, 124/9.741 đồ thị rỗng; CSQA dev 1.221 câu, 23 rỗng; OBQA train D_g 3.534 / D_v 2.000, 344/4.957 rỗng; OBQA test 500 câu, 38 rỗng | chưa công khai: [csqa-train](https://wandb.ai/duongkhang2100-university-of-science-ho-chi-minh-/graphml-seminar-report/runs/3ef5be158f64), [csqa-dev](https://wandb.ai/duongkhang2100-university-of-science-ho-chi-minh-/graphml-seminar-report/runs/4d16c1462dda), [obqa-train](https://wandb.ai/duongkhang2100-university-of-science-ho-chi-minh-/graphml-seminar-report/runs/7a0a5c6c575d), [obqa-test](https://wandb.ai/duongkhang2100-university-of-science-ho-chi-minh-/graphml-seminar-report/runs/6de9756cb81c) |
| G1 | Table 1: CSQA/OBQA — base, KAPING, GraphVis | | VII-B | G1a **xong** (lượt 1, 15/10/2026; số trong log trùng summary W&B): mô hình gốc CSQA dev text 70,93 / ảnh 68,06 / KAPING 68,39; OBQA test text 75,40 / ảnh 65,80 / KAPING 68,20 (bài: 70,5 / 67,7; 75,6 / 71,2). G1b **xong** (lượt 2, 16–17/10/2026): GraphVis CSQA dev 81,16 (bài 82,8); theo số nút 0-10: 85,71 / 11-20: 79,95 / 21-40: 80,62 / >40: 81,32; GĐ1 895 bước, 6 giờ 22 phút, loss 0,959; GĐ2 500 bước, 3 giờ 26 phút, loss 0,434; cả lượt ~10,4 giờ. G1c **xong** (lượt 8, 20/10/2026; GĐ1 6 giờ 27, GĐ2 3 giờ 35, cả lượt 10,3 giờ): GraphVis OBQA test 87,6 (bài 85,5); theo số nút 0-10: 86,07 / 11-20: 89,16 / 21-40: 87,38 / >40: 90,0 | chưa công khai: [csqa-text](https://wandb.ai/duongkhang2100-university-of-science-ho-chi-minh-/graphml-seminar-report/runs/011935dbe5a7), [csqa-image](https://wandb.ai/duongkhang2100-university-of-science-ho-chi-minh-/graphml-seminar-report/runs/5ea27e4e729f), [csqa-kaping](https://wandb.ai/duongkhang2100-university-of-science-ho-chi-minh-/graphml-seminar-report/runs/b069e0dae476), [obqa-text](https://wandb.ai/duongkhang2100-university-of-science-ho-chi-minh-/graphml-seminar-report/runs/f3d5f325e5ac), [obqa-image](https://wandb.ai/duongkhang2100-university-of-science-ho-chi-minh-/graphml-seminar-report/runs/536a9ff8edba), [obqa-kaping](https://wandb.ai/duongkhang2100-university-of-science-ho-chi-minh-/graphml-seminar-report/runs/fa55f17fd085), [G1b-gđ1](https://wandb.ai/duongkhang2100-university-of-science-ho-chi-minh-/graphml-seminar-report/runs/812759bf346e), [G1b-gđ2](https://wandb.ai/duongkhang2100-university-of-science-ho-chi-minh-/graphml-seminar-report/runs/9c7331e1dd49), [G1b-csqa-image](https://wandb.ai/duongkhang2100-university-of-science-ho-chi-minh-/graphml-seminar-report/runs/3522cf6077b0); cần gom vào một W&B Report công khai |
| G2 | Table 2: ScienceQA-IMG, MMBench, POPE ×3 | eval LLaVA | VII-C | **xong** (lượt 3, 17/10/2026; ~4,2 giờ). Gốc → GraphVis: SciQA-IMG 72,53 → 70,80; MMBench vòng tròn 65,00 → 47,75 (một lượt 74,78 → 64,77); POPE F1 r 83,33 → 76,10, p 81,55 → 76,28, a 80,90 → 74,46. **Ngược bài** (bài tăng ở cả 5). Chưa kiểm tra tỉ lệ không tách được đáp án | chưa công khai: [base-sciqa](https://wandb.ai/duongkhang2100-university-of-science-ho-chi-minh-/graphml-seminar-report/runs/d08e3ef2ed2e), [base-mmbench](https://wandb.ai/duongkhang2100-university-of-science-ho-chi-minh-/graphml-seminar-report/runs/75d667772b7f), [base-pope](https://wandb.ai/duongkhang2100-university-of-science-ho-chi-minh-/graphml-seminar-report/runs/2049ef03762e), [gv-sciqa](https://wandb.ai/duongkhang2100-university-of-science-ho-chi-minh-/graphml-seminar-report/runs/4fb1e66245dc), [gv-mmbench](https://wandb.ai/duongkhang2100-university-of-science-ho-chi-minh-/graphml-seminar-report/runs/83139d778b91), [gv-pope](https://wandb.ai/duongkhang2100-university-of-science-ho-chi-minh-/graphml-seminar-report/runs/ec8bcabbb71f) |
| G3 | Table 6: ScienceQA joint vs curriculum | | VII-C | **xong** (lượt 9, 20/10/2026; 56 phút): joint 70,70 (curriculum 70,80, gốc 72,53; bài 71,94 / 73,18) | [G3](https://wandb.ai/duongkhang2100-university-of-science-ho-chi-minh-/graphml-seminar-report/runs/ff41e4a42003) |
| G4 | Fig 7: joint / curriculum / graph-first / OCR-first | CSQA | VII-D | G4joint **xong** (lượt 4, 18/10/2026; train 10,1 giờ, cả lượt 10,7 giờ): 80,34 (bài 78,05). G4graph: train xong cả 3 giai đoạn (4 giờ 42 + 2 giờ 49 + 4 giờ 04, adapter trên Hub), đánh giá bị cắt ở mốc 12 giờ (lượt 5) → chạy lại lượt 5: **80,84** (bài 81,00). G4ocr **xong** (lượt 6; 2 giờ 27 + 4 giờ 07 + 3 giờ 31, cả lượt 10,7 giờ): **81,74** (bài 82,80). Thứ tự OCR trước > curriculum > cấu trúc trước > joint giống bài, khoảng chênh 1,4 điểm (bài 4,75) | chưa công khai: [joint-train](https://wandb.ai/duongkhang2100-university-of-science-ho-chi-minh-/graphml-seminar-report/runs/e2363830f9e4), [joint-eval](https://wandb.ai/duongkhang2100-university-of-science-ho-chi-minh-/graphml-seminar-report/runs/179636dc91ae), [graph-1a](https://wandb.ai/duongkhang2100-university-of-science-ho-chi-minh-/graphml-seminar-report/runs/34a92ee1c2df), [graph-1b](https://wandb.ai/duongkhang2100-university-of-science-ho-chi-minh-/graphml-seminar-report/runs/9fa39e904198), [graph-2](https://wandb.ai/duongkhang2100-university-of-science-ho-chi-minh-/graphml-seminar-report/runs/1d1796a1dc88) |
| G5 | Table 3: ±KAPING trên base / joint / curriculum | CSQA | VII-D | **xong** (lượt 9; chế độ image+kaping): gốc 65,77 (chỉ ảnh 68,06), joint 80,02 (80,34), curriculum 79,36 (81,16); bài 67,65 / 78,71 / 81,74 | [base](https://wandb.ai/duongkhang2100-university-of-science-ho-chi-minh-/graphml-seminar-report/runs/999775073052), [joint](https://wandb.ai/duongkhang2100-university-of-science-ho-chi-minh-/graphml-seminar-report/runs/49bd87b7a621), [curriculum](https://wandb.ai/duongkhang2100-university-of-science-ho-chi-minh-/graphml-seminar-report/runs/1f2f6749d61b) |
| G6 | Table 4: 6 tác vụ đọc hình trước/sau GĐ1 (exact match) | test CSQA | VII-E | **xong** (lượt 3; ~2,5 giờ; 600 câu, ~100 ảnh CSQA dev). Gốc → sau GĐ1 (%): tên nút 9,96 → 21,09; bậc 9 → 55; bậc max 1,01 → 9,09; số nút 0 → 11; số cạnh 1 → 14; bộ ba 0,48 → 6,97 | chưa công khai: [base](https://wandb.ai/duongkhang2100-university-of-science-ho-chi-minh-/graphml-seminar-report/runs/09dcc3de6354), [after-stage1](https://wandb.ai/duongkhang2100-university-of-science-ho-chi-minh-/graphml-seminar-report/runs/8c408d5c80ce) |
| G7 | Fig 9: ảnh 50×50 | CSQA | VII-E | **xong** (lượt 7: lần đầu dừng giữa GĐ2 vì hết quota, GĐ1 6 giờ 32 đã lên Hub; lần chạy lại GĐ2 4 giờ 13 + đánh giá 33 phút): 80,26 (G1b 81,16; bài ~77–78, ước lượng từ biểu đồ) | [eval](https://wandb.ai/duongkhang2100-university-of-science-ho-chi-minh-/graphml-seminar-report/runs/363d62024512) |
| C1 | Độ chính xác theo số nút/cạnh (≤10, 11–20, 21–40, >40) | base, GraphVis, C3 | VI-C | **xong** (lượt 9): bảng 5 mô hình × 4 nhóm số nút (gốc text/ảnh, GraphVis, C3, G7); không nhóm nào GraphVis hơn C3 có ý nghĩa | [C1](https://wandb.ai/duongkhang2100-university-of-science-ho-chi-minh-/graphml-seminar-report/runs/0c067d482176) |
| C2 | Độ chính xác theo độ phân giải (50/112/224/336/672, AnyRes) | | VI-C | **xong** (lượt 9 hỏng ở graphvis-anyres-672 vì hết VRAM, chạy lại 2 điểm sau PR #13; 300 câu/điểm): GraphVis 81,7–82,7 ở mọi kích thước và chế độ; gốc 71,0 ở 50 px, 64,3–67,0 ảnh gốc | 24 run `C2-*` trong report VI-C |
| C3 | Đối chứng: tinh chỉnh CSQA **không ảnh**, cùng siêu tham số | | VI-C | **xong** (lượt 3; train 26 phút, loss 0,521): CSQA dev 80,59 (GraphVis 81,16, gốc 70,93); theo số nút 0-10: 86,86 / 11-20: 77,22 / 21-40: 81,59 / >40: 79,12; 1 câu không tách được đáp án | chưa công khai: [train](https://wandb.ai/duongkhang2100-university-of-science-ho-chi-minh-/graphml-seminar-report/runs/395d28b8ff1e), [eval](https://wandb.ai/duongkhang2100-university-of-science-ho-chi-minh-/graphml-seminar-report/runs/4c3616154742) |

Phân tích lại từ bảng predictions (18/10/2026, `repro/graphvis/scripts/analyze_predictions.py`): GraphVis so với C3 62/55 câu, McNemar p=0,58; curriculum so với joint 82/72, p=0,47; gốc text so với gốc ảnh 119/84, p=0,017; MMBench gốc so với GraphVis 231/83, p<0,001; ScienceQA 197/162, p=0,07. G6 gốc từ chối đếm ở 92/100 (nút) và 89/100 (cạnh) câu. POPE không lưu bảng dự đoán.

**Báo cáo W&B công khai (21/10/2026).** Project nằm trong team của gói Academic trial, chỉ để được chế độ Team. Đã tắt tùy chọn team *Disable public sharing to outside of team for reports in private projects* (Khang đồng ý) để report dùng view-only link: ai có link xem được, không cần đăng nhập; run và project vẫn riêng tư. Mỗi mục một report, đã gắn vào báo cáo bằng `\wandbreport{}`:

- VI-C (C1, C2, C3): [VI-C](https://forge.coreweave.com/wandb/duongkhang2100-university-of-science-ho-chi-minh-/graphml-seminar-report/reports/VI-C-ph-c-t-p-nh-th-C1-C2-C3---VmlldzoxODE1MDkyMg?accessToken=k0p9ayvl7n6xu0nxh4m064aw1i7eic2rd5wk7fnzai39y1pqbugkmwx6sjnr9ree)
- VII-B (G1a, G1b, G1c): [VII-B](https://forge.coreweave.com/wandb/duongkhang2100-university-of-science-ho-chi-minh-/graphml-seminar-report/reports/VII-B-K-t-qu-ch-nh-CSQA-OBQA---VmlldzoxODE1MDg3OQ?accessToken=utsf3q1euuqow02lp5ngsc484lljatsdpjw3ifo7gms5oy3gkwpyqju4vxlt6hmh)
- VII-C (G2, G3): [VII-C](https://forge.coreweave.com/wandb/duongkhang2100-university-of-science-ho-chi-minh-/graphml-seminar-report/reports/VII-C-Chuy-n-giao-sang-VQA-G2-G3---VmlldzoxODE1MDk3NA?accessToken=rr16q2af8ndn3u4e9xfn6tt5jotymrtmnkrfgzq9t3xqzmzkfzw1tt9qowp5wmix)
- VII-D (G4, G5 (kèm G1b)): [VII-D](https://forge.coreweave.com/wandb/duongkhang2100-university-of-science-ho-chi-minh-/graphml-seminar-report/reports/VII-D-Ablation-th-t-hu-n-luy-n-v-KAPING--VmlldzoxODE1MDkxMw?accessToken=x21tw6skdy91vcuqe2zj7ii3phsovsxuzqjmffp97a2xt7jq4021rjvefbojeq81)
- VII-E (G6, G7 (kèm G1b)): [VII-E](https://forge.coreweave.com/wandb/duongkhang2100-university-of-science-ho-chi-minh-/graphml-seminar-report/reports/VII-E-c-nh-th-v-nh-50x50-G6-G7---VmlldzoxODE1MDk0Nw?accessToken=u86b4vbk27j2qmbngzw095f0sapohlzdrql3htt2ix8qz06ae8azzbie3x4o1nig)

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
| Ex01 | Chi phí toàn bài `nervous-system` | **Xong, dùng trong báo cáo:** lượt `gpt-6-luna` chạy đủ mọi bước (14 đoạn, 27 ảnh): 335 lượt gọi, không có lần thử lại; 4.325.932 token vào, trong đó 2.779.392 đã lưu sẵn; 14.909 token ra; khoảng 32 phút. Một bài khoảng 0,027–0,19 USD theo giá tra ngày 14/10/2026: cận dưới theo token kiểu API, cận trên theo token CLI. Video/PDF gốc và tiền xử lý lịch sử còn thiếu; chất lượng chưa chấm. Lượt 13/10 đã xóa khỏi máy. | Chưa kiểm tra công khai: [run gpt-6-luna](https://wandb.ai/nhinbm-hcmus/SciMKG/runs/7b20992d11ea) |
| Ex02 | Sáu bài × ba cấu hình: 0 vòng, 1 vòng, 5 vòng kiểm chứng (ngưỡng 0,7), chấm sau Integration, trước Augmentation | **Xong, dùng trong báo cáo:** 18 lượt; F1 trung bình 0,392 (0 vòng), 0,362 (1 vòng), 0,373 (5 vòng); lượt 5 vòng trên `nervous-system` dùng lại Ex01 | [ex02-summary](https://wandb.ai/nhinbm-hcmus/SciMKG/runs/ex02-8319b815) (công khai) |
| Ex03 | Tìm đúng đoạn bài giảng trên 18 lượt Ex02: 309 câu nguyên văn và 309 câu viết lại, gắn khái niệm theo đoạn đầu hoặc mọi đoạn có nhắc | **Xong, dùng trong báo cáo:** Hit@1 nguyên văn 0,657 (mọi đoạn, 0 vòng), 0,483 (đoạn đầu, 0 vòng), BM25 1,000 (mốc trên), ngẫu nhiên 0,085 | [ex03-gpt-6-luna](https://wandb.ai/nhinbm-hcmus/SciMKG/runs/krpbt4zl) (công khai) |

Dữ liệu: [manifest](../SciMKG/data/MOOC/manifest.json). Nguồn, danh sách và giới hạn: [MOOC](../SciMKG/docs/MOOC.md). Lệnh chuẩn bị/kiểm tra: `python3 scripts/preprocessing/prepare_mooc.py` / `--check` trong fork. Hai bài Vật lý dự kiến ban đầu bị cắt dở trong nguồn nên đã thay trước khi chạy mô hình bằng áp suất khí quyển và lực đẩy. Bộ nhỏ này chưa đại diện toàn tập; chất lượng Augmentation chưa có nhãn tương ứng.

Báo cáo Ex01 của lượt 13/10 (đã thay bằng `gpt-6-luna`, không còn dùng trong báo cáo) trên mục **Reports**: [Token và chi phí một bài học](https://wandb.ai/nhinbm-hcmus/SciMKG/reports/Ex01-Token--VmlldzoxODEwMzc1NA==), công khai đã kiểm tra không đăng nhập. Mở đầu giải thích đầu vào, kết quả và giá API giả định; bảng kỹ thuật được thu gọn. Có tổng 5.837.546 token vào, 39.332 token ra, bảng từng bước và biểu đồ hai nhánh; 1.878.144 token cache dịch vụ đã nằm trong đầu vào. Báo cáo giữ riêng chi phí phát sinh và dựng riêng giả định, ghi rõ 9 lần thiếu usage. Tạo từ nhật ký cũ, không thêm suy luận.

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

Quyết định mới cho lần chạy thử này: không dùng mã PR #1, mẫu nhỏ tác giả trước và một bài tiếng Việt sau, Codex dùng thuê bao. Báo cáo hợp nhất: pilot-report.html (đã gỡ khỏi repo, xem commit `bc2af4e`). Mã và lệnh chạy: pilot/README.md (đã gỡ khỏi repo, xem commit `bc2af4e`).

| ID | Công việc | Trạng thái | W&B |
|---|---|---|---|
| P0 | Đối chiếu bài ↔ mã gốc, triển khai và kiểm thử độc lập | Đối chiếu trực tiếp 11 tệp; **38 kiểm thử CPU đạt**. Dữ liệu kiểm thử giả lập; đối chiếu lượt thật ghi riêng ở P1/P2. | Ngoại tuyến |
| P1 | Kiểm tra quy trình trên một mẫu nhỏ của tác giả | Đã tải mẫu đúng hash; lấy thêm chỉ thành phần một bài qua Range. Chạy hai đoạn 06:35–08:07: 17 khái niệm + 2 bổ sung, 61 nút/132 cạnh, 19 WAV; 46 lời gọi thật. Thiếu video; nhãn phạm vi chưa duyệt. | Ngoại tuyến `vl2j1ruo` |
| P2 | Kiểm tra quy trình trên một bài tiếng Việt | Sau P1, chạy hai đoạn OLM 01:14–02:03: 10 khái niệm + 1 bổ sung, 38 nút/108 cạnh, 11 WAV; 36 lời gọi thật. Clip 49 giây đã kiểm tra tệp/mốc; nhãn Codex gán tạm, nội dung chờ duyệt. | Ngoại tuyến `amwom3iv` |
| P3 | Hai người chấm nội dung, nghe âm thanh, công khai kết quả | **Chờ người duyệt và video đúng bài tác giả**. Phần ghi nhận xét (đã gỡ khỏi repo, xem commit `bc2af4e`) có ảnh/âm thanh/clip Việt và bản dịch hỗ trợ đọc mẫu tác giả; W&B SDK lưu hai lượt ngoại tuyến, chưa có dự án/link công khai. | Chưa công khai |

Không dùng các kết quả chấm tác giả trước đây làm kết quả mới của lần chạy thử này. Các mục S0–S5 phía trên là kế hoạch tái lập cũ, chưa phải các lượt đã chạy của P0–P3.

Cập nhật 11/10/2026: tổng **82 lời gọi thật**, 1.169.622 token đầu vào, 8.155 token đầu ra, 641,66 giây tổng thời gian CLI; token có ngữ cảnh hệ thống. Dùng xác thực ChatGPT; tiền API chưa đo. Chạy lại cache không thêm suy luận và không đổi hash JSON/RDF. P/R/F1 hiện chỉ là phép chấm trên nhãn tạm, **không phải kết quả chất lượng chính thức**. Chưa có thí nghiệm loại bỏ thành phần độc lập; cả bốn đoạn giữ nguyên ứng viên qua năm vòng. Bằng chứng/hash và các hạn chế ở báo cáo pilot (đã gỡ khỏi repo, xem commit `bc2af4e`).

Phần duyệt mẫu tác giả dùng bản dịch tiếng Việt do Codex tạo; kiểm chứng bản gốc và phát âm tiếng Trung cần người biết ngôn ngữ. JSON nhận xét ghi rõ căn cứ đọc. Dữ liệu và các số đo P0–P2 giữ nguyên.

Đã bổ sung bảng đối chiếu đầu ra tác giả (đã gỡ khỏi repo, xem commit `bc2af4e`) và bằng chứng nguồn/mã băm, không phát sinh lời gọi LLM. Khớp 5/6 tên chưa chứng minh mọi dự đoán đúng. Tính lại phiếu của hai đoạn giữ 3 khái niệm, còn tệp tích hợp công bố ghi 1; cấu hình sinh chưa rõ. Mẫu ti thể có tài nguyên tham chiếu, chưa phải nhãn chuẩn đa phương thức của đúng trích đoạn. Không yêu cầu Nhi tự chấm chuyên môn chưa có; phần chưa thể xác minh ghi rõ chưa chấm.

Trước khi chạy dữ liệu thực sau pilot: bổ sung và kiểm chứng lệnh chuẩn bị đầu vào từ video/PDF (âm thanh, bản chép lời có thời gian, phân đoạn, ảnh, clip). Theo người dùng ngày 11/10/2026, việc này nằm sau pilot; hiện giữ nguyên mã và dữ liệu của lượt đã chạy. Xem điều kiện ở bước 01 (đã gỡ khỏi repo, xem commit `bc2af4e`).

Đồng bộ EVIA ngày 11/10/2026: pilot gọi trực tiếp mã fork qua `main_evia.py`; chạy `python audit_run.py --check-main-evia` để đối chiếu phản hồi cũ trên mẫu tác giả trước, tiếng Việt sau. Đã kiểm tra cờ nguồn, ngưỡng/lỗi và JSON/RDF; không gọi LLM mới, không thay đầu ra lịch sử. Lượt mới lưu ở `runs/main-evia/`. ConceptNet còn thiếu luật, API Wikipedia trực tiếp chưa kiểm chứng; mục đồng bộ và bằng chứng E11 (đã gỡ khỏi repo, xem commit `bc2af4e`).
