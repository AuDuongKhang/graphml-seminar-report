# CLAUDE.md — Hướng dẫn cho agent (Claude Code) làm tiếp báo cáo này

Đọc file này trước tiên, sau đó đọc `docs/context.md`. Không cần đọc lại hai bài báo từ đầu:
Phát hiện GraphVis: `docs/findings.md`; SciMKG: đọc `SciMKG/docs/FINDINGS.md`.

## 1. Dự án là gì

Báo cáo giữa kỳ môn **Học máy trên đồ thị** (Thạc sĩ, ĐH KHTN – ĐHQG-HCM, GV **Lê Ngọc Thành**).
Nhóm 2 người:

| Thành viên | MSHV |
|---|---|
| Âu Dương Khang | 25C15045 |
| Nguyễn Bùi Mẫn Nhi | 25C15055 |

Một báo cáo IEEE duy nhất, tiếng Việt, gồm hai phần **độc lập** (hai bài được chọn ngẫu nhiên, **không** ép liên kết):

- **Phần A — GraphVis** (NeurIPS 2024): bài nhóm đã seminar. Vẽ đồ thị con ConceptNet thành ảnh, cho LLaVA-1.6 đọc; huấn luyện 2 giai đoạn (đọc ảnh đồ thị → trả lời QA).
- **Phần B — SciMKG** (AAAI 2026): bài "chỉ báo cáo". Dựng tự động KG giáo dục đa phương thức (văn bản/ảnh/video/âm thanh) bằng chuỗi nhiều LLM (EVIA). **Không huấn luyện mô hình nào** — chỉ gọi API.

File nộp: `25C15045-25C15055.pdf`, ≤ 25 trang (tính cả bìa, tham khảo, phụ lục), < 20 MB.

## 2. Cấu trúc repo

```
CLAUDE.md               ← file này
AGENTS.md               ← trỏ về file này (cho Codex/Cursor/agent khác)
README.md               ← hướng dẫn CHẠY tái lập (yêu cầu môn học; KHÔNG ghi kết quả/giải pháp ở đây)
docs/
  context.md            ← yêu cầu môn học, nhận xét của GV, quyết định đã chốt
  findings.md           ← phát hiện GraphVis; SciMKG ở SciMKG/docs/FINDINGS.md
  experiments.md        ← danh sách thực nghiệm cần tái lập, trạng thái, chỗ dán link W&B
  plan_bao_cao.md       ← kế hoạch & sườn ban đầu (lịch sử; ưu tiên các file trên khi mâu thuẫn)
report/                 ← mã LaTeX của báo cáo
  main.tex              ← preamble, macro, thứ tự \input
  sections/00_cover.tex,
           A_intro.tex, A_background.tex, A_graphvis.tex, A_conclusion.tex (Phần A),
           B_intro.tex, B_related.tex, B_scimkg.tex, B_conclusion.tex (Phần B), B_tables_parked.tex (bảng Phần B tạm gỡ, không `\input`),
           04_contributions.tex, 99_appendix.tex
  figures/*.tex         ← hình TikZ (vẽ lại, không chụp hình bài báo)
  tables/*.tex          ← bảng sinh tự động từ số đo (không sửa tay)
  tools/make_ex_tables.py ← sinh các bảng số đo của Phần B vào report/tables/ (ex01_cost, ex02_quality, ex02_summary, ex03_retrieval) từ SciMKG/runs/analysis; chạy lại sau analyze.py (Ex01), run.py summary (Ex02), run.py evaluate (Ex03)
  refs.bib
GraphVis-code/          ← mã gốc GraphVis (Apache-2.0). Commit đầu tiên = bản gốc chưa sửa;
                          mọi sửa đổi để tái lập xem bằng `git diff <commit đầu> -- GraphVis-code`
SciMKG/                 ← fork nhinbm/SciMKG; phát triển và kiểm thử tại đây
repro/                  ← common/ và graphvis/ (package gv, README, DEVIATIONS, notebook)
```

Hướng dẫn `smkg` trong `docs/` là bản cũ; mã SciMKG hiện nằm trong fork.

Không có trong git (chỉ có trên máy Khang): các file `.txt` ngữ cảnh (yêu cầu, nhận xét GV,
link slide Canva), PDF hai bài báo. Nội dung cần thiết đã được tóm tắt vào `docs/context.md`.

## 3. Biên dịch báo cáo

```bash
cd report
latexmk -xelatex main          # hoặc: xelatex main && bibtex main && xelatex main && xelatex main
```

- **Bắt buộc XeLaTeX** (fontspec + babel vietnamese). Font chính `Times New Roman` (có sẵn trên Windows).
  Trên Linux/Overleaf: đổi `\setmainfont{Times New Roman}[NFSSFamily=ptm]` thành font có tiếng Việt
  (vd. `TeX Gyre Termes` nếu đã cài, hoặc `Liberation Serif`), giữ `NFSSFamily=ptm`.
- Cảnh báo `Font shape TU/ptm/... undefined` là vô hại (IEEEtran gọi ptm trước fontspec).
- Sau khi biên dịch, kiểm tra: `grep -c "^!" main.log` = 0, không có `Overfull`, số trang ≤ 25
  (`pdfinfo main.pdf | grep Pages`).
- Không dùng ký tự CJK (tiếng Trung) trong .tex — font không có glyph. Ghi nghĩa tiếng Việt.
- Trên Git Bash/Windows: heredoc + Python có thể làm mất dấu `\` trong chuỗi LaTeX. Sửa file .tex
  bằng công cụ Edit, hoặc viết script Python ra file rồi chạy.

## 4. Quy tắc viết (bắt buộc)

1. **Chỉ tiếng Việt**, không trộn tiếng Anh ngoài thuật ngữ/tên riêng (thuật ngữ lần đầu có thể kèm tiếng Anh).
2. **Không chép câu chữ bài gốc** — diễn đạt lại bằng lời nhóm. Hình phải vẽ lại (TikZ).
3. **Không bịa số liệu.** Ô chưa có kết quả để `\todo{...}`. Số đọc từ biểu đồ phải ghi "ước lượng từ biểu đồ".
   Ví dụ minh họa do nhóm dựng phải ghi rõ "ví dụ minh họa, không phải số liệu bài".
4. Nhận xét của GV (xem `docs/context.md` §2) được **lồng vào như thông tin chi tiết hơn** —
   **không** viết "trả lời nhận xét của thầy", "slide 3", "slide 11"... trong báo cáo.
5. **Không ép liên kết** giữa GraphVis và SciMKG.
6. Phần SciMKG phải chi tiết ngang GraphVis: người đọc hình dung được hệ thống chạy thế nào
   (dữ liệu vào/ra, mô hình ở mỗi bước, prompt làm gì, cách tính điểm).
7. Mỗi mục thực nghiệm gắn **một link W&B công khai riêng** (macro `\wandb{url}`; chỗ trống dùng `\wandbtodo`).
8. Bảng 1 cột quá rộng: bọc `\resizebox{\columnwidth}{!}{...}`; bảng nhỏ thì không bọc (sẽ bị phóng to).

Macro có sẵn trong `main.tex`: `\todo{}`, `\wandb{}`, `\wandbtodo`, `\GV`, `\SM`, `\Dg`, `\Dv`, `\phan{}{}`.
Trước khi nộp: `grep -rn "\\\\todo\|wandbtodo" report/` phải rỗng.

## 5. Trạng thái hiện tại (cập nhật khi làm xong việc)

| Phần | Trạng thái |
|---|---|
| Bìa, tiêu đề, tác giả | Xong (còn link GitHub/W&B trên bìa) |
| Cấu trúc báo cáo (12/10/2026) | Hai phần độc lập, mỗi phần có giới thiệu, kiến thức nền và kết luận riêng; không còn giới thiệu/kiến thức nền/kết luận chung. Đã bỏ tóm tắt và từ khóa chung; sau tiêu đề/tác giả vào thẳng Phần A. Phần B mở ở trang mới; mỗi phần đánh số mục lại từ I (Phần A I–IX; Phần B I–V: Giới thiệu, Công trình liên quan, Phương pháp, Thực nghiệm, Kết luận). Đã biên dịch XeLaTeX (14/10/2026, sau khi viết Giới thiệu, Công trình liên quan, Phương pháp và Thực nghiệm của Phần B): **14 trang**, không lỗi, không tràn lề, không tham chiếu chưa giải quyết. |
| A: I Giới thiệu | Đã viết đoạn giới thiệu ngắn |
| B: I Giới thiệu (13/10/2026) | Bản nháp đầy đủ: bối cảnh, hai hạn chế (có trích dẫn KnowEdu, MOOCCubeX, Jhajj, TIVA-KG), ý tưởng và số liệu theo bài, đóng góp của báo cáo, lộ trình. Link bài và mã nằm trong danh mục tham khảo, không dùng chú thích. Lộ trình đã có `\ref` tới bốn mục còn lại (14/10/2026). |
| A: II Kiến thức nền | **Sườn** |
| B: II Công trình liên quan (13/10/2026) | Bản nháp đầy đủ trong `B_related.tex` (thay `B_background.tex`): 5 mục con, mỗi mục một đoạn, bảng so sánh đồ thị đa phương thức tạm gỡ vào `B_tables_parked.tex`. Mô tả công trình lấy từ tóm tắt hoặc trang dự án, chưa đọc toàn văn. |
| A: III–VI, VIII (GraphVis) | Bản nháp hoàn chỉnh |
| A: VII Tái lập; VI-C thực nghiệm độ phức tạp ảnh | **Chờ chạy** (bảng `\todo`) |
| B: III Phương pháp (13/10/2026) | Đã viết lại ngắn gọn, ưu tiên nội dung bài báo; chi tiết rút từ mã nguồn và các điểm không khớp gom ở III-F (đối chiếu commit `81f2ce2`). Hình tổng quan dùng nguyên Hình 2 của bài (cắt vector từ PDF, có ghi nguồn), khác quy tắc mục 4 "Hình phải vẽ lại": cần Khang đồng ý. Các bảng của Phần B tạm gỡ vào `report/sections/B_tables_parked.tex` (không `\input`). |
| B: IV Thực nghiệm (15/10/2026) | Viết theo Ex01 đến Ex03 đã xong: A kết quả công bố (mốc), B thiết lập (hai cách đếm token, cách đo chất lượng, sửa điều kiện tích hợp), C Ex01 (chi phí một bài), D Ex02 (0, 1, 5 vòng kiểm chứng), E Ex03 (tìm đúng đoạn, mới), F thảo luận. Bảng sinh bằng `report/tools/make_ex_tables.py` vào `report/tables/`. Không còn `\todo` trong Phần B. |
| B: V Kết luận (15/10/2026) | Đã viết: chi phí một bài, kiểm chứng và truy hồi, điểm không khớp giữa bài báo và mã, giới hạn, hướng phát triển. |
| SciMKG: lần chạy thử độc lập tại `papers/SciMKG/pilot/` (10–11/10/2026) | **38 kiểm thử CPU đạt**; đã chạy hai đoạn tác giả trước, hai đoạn tiếng Việt sau; 82 lời gọi thuê bao, JSON/RDF, 30 WAV và clip Việt 49 giây đã kiểm tra kỹ thuật. **Chờ duyệt nhãn/nội dung; thiếu video tác giả, W&B mới ngoại tuyến.** Báo cáo hợp nhất: `papers/SciMKG/pilot/report/pilot-report.html` (kế hoạch, phần duyệt có bản dịch tiếng Việt cho mẫu tác giả, kết quả, đối chiếu mã và nguồn). Không dùng mã PR #1. |
| A: IX Kết luận; Phân công, Phụ lục B–C | **Sườn** |
| Code tái lập | GraphVis ở `repro/graphvis/`; SciMKG phát triển trong fork `SciMKG/`. |
| Pilot EVIA đồng bộ với fork (11/10/2026) | Gọi trực tiếp các hàm EVIA qua `main_evia.py`; đối chiếu phản hồi lưu sẵn trên mẫu tác giả rồi tiếng Việt, đủ bốn tổ hợp cờ nguồn, không gọi LLM mới. Lượt cũ giữ nguyên; xem mục `#main-evia-check` trong báo cáo pilot. Sửa mã chính và changelog tương thích pilot hiện còn cục bộ, chưa commit. |
| SciMKG: chuẩn bị chung Ex01/Ex02 (13/10/2026) | **Xong dữ liệu:** 6 bài MOOC có nhãn tác giả, 75 đoạn và 188 nhãn theo bài; `SciMKG/data/MOOC/manifest.json` cố định nguồn/CRC/SHA-256, đầu vào và nhãn tách riêng. Ex01 dùng `nervous-system`, Ex02 dùng cả 6 bài × 3 cấu hình. Hai bài Vật lý cắt dở trong nguồn đã thay bằng áp suất khí quyển và lực đẩy. **24 kiểm thử CPU đạt**, chạy lại giữ nguyên 19 hash; chưa gọi LLM/chạy Ex01–Ex02. Xem `SciMKG/docs/MOOC.md`. |
| SciMKG: Ex01 lượt 13/10/2026 (đã thay bằng gpt-6-luna, 14/10/2026) | **Xong trong phạm vi văn bản/ảnh:** 14 đoạn, 27 ảnh; hai nhánh dùng chung 55 khái niệm E/V/I. Nhánh ConceptNet thêm 40 khái niệm, 99 giải thích/WAV, 319 nút/623 cạnh. Tổng hai nhánh 436 lần thử; phần usage đã biết tương đương API **8,499938–10,479639 USD**, 9 lần thiếu usage, dùng thuê bao Codex và TTS cục bộ. **29 kiểm thử CPU đạt**, replay 427 cache không suy luận, 1.762 trace đã đối chiếu. Hai run W&B Public đã kiểm tra không xác thực; xem `SciMKG/docs/EX01.md`. Video/PDF gốc, tiền xử lý lịch sử còn thiếu; chất lượng chưa chấm. Ex02 chưa chạy; số liệu Ex01 đã đưa vào LaTeX (Mục IV-C Phần B). |
| SciMKG: báo cáo token Ex01 lượt 13/10/2026 (đã thay bằng gpt-6-luna) | **Public, đã đọc không đăng nhập:** [Token và chi phí](https://wandb.ai/nhinbm-hcmus/SciMKG/reports/Ex01-Token--VmlldzoxODEwMzc1NA==). Phần đầu giải thích đầu vào, kết quả, token và giá API giả định; bảng kỹ thuật thu gọn. Tổng 5.837.546 token vào, 39.332 token ra; 1.878.144 token cache đã nằm trong đầu vào, 9 lần thiếu usage. Có bảng từng bước và biểu đồ hai nhánh; tạo báo cáo không thêm suy luận. |
| SciMKG: tổ chức thư mục Ex01 (13/10/2026) | Dữ liệu sinh ra chuyển từ `data/ex01/` sang `runs/ex01/preparation/`; toàn bộ `runs/` ignored, chưa có tệp được Git theo dõi. Mã chuyển sang `experiments/ex01/`, cần commit; `scripts/run_ex01.py` chỉ là entry point 11 dòng, CLI ở `experiments/ex01/cli.py`, đồng bộ W&B ở `tracking.py`. E/V/I gọi mã gốc, các bước bổ sung/căn chỉnh/TTS/xuất đồ thị có triển khai riêng. Theo người dùng đã gỡ cả test và script audit hiện hành; snapshot/bằng chứng lịch sử giữ nguyên. |
| SciMKG: Ex02 và Ex03 (15/10/2026) | **Xong, đã publish:** Ex02 gồm 18 lượt (3 cấu hình × 6 bài), F1 trung bình 0,392 (không kiểm chứng), 0,362 (1 vòng), 0,373 (5 vòng); chưa thấy kiểm chứng làm tăng F1, còn 5 vòng tốn gấp khoảng 4 lần. Ex03 (tìm đúng đoạn bài giảng): Hit@1 nguyên văn 0,657 khi gắn mọi đoạn có nhắc, không kiểm chứng; cách của tác giả (chỉ đoạn đầu) 0,483; BM25 1,000 (mốc trên). Run W&B tổng hợp: [Ex02](https://wandb.ai/nhinbm-hcmus/SciMKG/runs/ex02-8319b815), [Ex03](https://wandb.ai/nhinbm-hcmus/SciMKG/runs/krpbt4zl). Đã sửa điểm Ex03 phụ thuộc hạt giống; Đã xóa 17 run lẻ của Ex02 (15/10/2026); còn run Ex01 `7b20992d11ea` và hai run tổng hợp. |
| Chạy thực nghiệm trên Kaggle/Colab, dán link W&B | **Việc tiếp theo** — xem `docs/experiments.md` |
| SciMKG: CLI Ex01 rút gọn (13/10/2026) | Chỉ giữ `run` và `report`; bỏ `prepare`, `check`, `preprocess` cùng argument/import riêng. `run` tự chuẩn bị bài hệ thần kinh nếu thiếu đầu vào mặc định; pipeline/tracking kiểm tra dữ liệu và đăng nhập. Không chạy suy luận mới, không thêm test. |
| SciMKG: rà soát mã Ex01 sau chạy (13/10/2026) | Replay 427 cache lịch sử không suy luận; hash journal/graph giữ nguyên. Sửa chuẩn hóa chữ (17/40 nhãn bổ sung trùng E/V/I sau chuyển chữ), bí danh, hết hạn mức, ghi ngắt, kiểm tra cache, tính thời gian/giá, offline và trạng thái từng bước. Mã/prompt kỹ thuật tiếng Anh; protocol 2 dành cho lượt mới. Rà 23 tên argument CLI, bỏ argument xuất đồ thị và bốn import dư. **Người dùng yêu cầu gỡ test Ex01 và pytest; đã gỡ, không thêm lại.** Không chạy lại Ex01 hoặc sửa Workspace/Report W&B; chi tiết trong `SciMKG/docs/EX01.md`. |
| SciMKG: Ex01 dùng lại mã tác giả (14/10/2026) | Protocol 3; đã chạy đủ mọi bước với gpt-6-luna, xem dòng dưới. Bổ sung A gọi `NER.conceptExpand`/`ConceptNet.py` của tác giả; API ConceptNet lỗi 502 nên trả lời từ dump 5.7.0 (ghim SHA-256, chỉ mục SQLite 3,7 GB tại `SciMKG/runs/preprocessing/conceptnet/`, không vào Git). `current` giữ luật rỗng, `completed` chỉ điền luật mẫu câu tiếng Anh. Ảnh, giải thích dùng prompt tác giả; RDF qua `RDF/rdf.py`. Gỡ mã thay thế và mã chết. Chạy thử ngoại tuyến bằng client giả đạt; lượt thật cần lời gọi thuê bao mới. Xem `SciMKG/docs/EX01.md`, phát hiện F2 trong `SciMKG/docs/FINDINGS.md`. `runs/` tổ chức lại: `preprocessing/`, `experiments/ex-01/<tham số>/`, dữ liệu cũ ở `archived/ex-01/` chờ quyết định xóa; `scripts/` xếp giống hệt (`scripts/experiments/ex-01/run.py`). |
| SciMKG: Ex01 bằng gpt-6-luna (14/10/2026) | Lượt chạy đủ mọi bước trên bài Cấu tạo hệ thần kinh: 335 lượt gọi, 4.325.932 token vào (2.779.392 đã lưu sẵn), 14.909 token ra, khoảng 32 phút. Mục IV-C và `report/tables/ex01_cost.tex` lấy từ lượt này; giá một bài 0,027–0,19 USD (cận dưới theo token kiểu API, cận trên theo token CLI; công thức ở Mục IV-C). Đã xóa lượt 13/10 và lượt so sánh thứ hai khỏi `runs/`. Run W&B `7b20992d11ea`, chưa kiểm tra công khai. |

Việc tiếp theo với **lần chạy thử SciMKG của Nhi**: đọc đầu ra và bảng đối chiếu tác giả tại `papers/SciMKG/pilot/report/pilot-report.html#review`; chất lượng nhãn/nội dung/ngữ âm chưa có người đủ chuyên môn chấm phải ghi rõ chưa chấm, không yêu cầu Nhi tự xác nhận. Tìm video tác giả đúng bài; cấu hình dự án W&B công khai rồi đồng bộ lượt ngoại tuyến. Đã chạy đúng thứ tự tác giả trước, Việt sau; chưa phải toàn bài/tập đầy đủ. Tệp phiếu và tích hợp tác giả chưa khớp khi tính lại cho hai đoạn (3 so với 1 khái niệm), chưa rõ cấu hình sinh. Không dùng mã PR #1 hoặc tự chuyển sang API trả phí. Quyết định này thay cấu hình tái lập cũ trong phạm vi lần chạy thử; xem `docs/context.md` §5. Các mục S0–S5 trong `docs/experiments.md` chờ triển khai lại trong fork SciMKG.
Kết quả chấm tác giả trước đây giữ trong lịch sử Git; phát hiện SciMKG đang duyệt ở `SciMKG/docs/FINDINGS.md`.
Sai khác GraphVis: `repro/graphvis/DEVIATIONS.md`; SciMKG: ghi tại fork.

## 6. Làm việc nhóm (2 người, 2 agent)

- Mỗi người làm trên **nhánh riêng** (`khang/...`, `nhi/...`), mở PR vào `main`; `git pull --rebase` trước khi bắt đầu.
- Tránh sửa cùng một file cùng lúc. Phân chia gợi ý: một người giữ `sections/A_graphvis.tex` + `repro/graphvis/`,
  người kia giữ `sections/B_scimkg.tex` + fork `SciMKG/`. `main.tex`, `refs.bib` sửa nhỏ, báo trước.
- Khi xong một việc: cập nhật bảng §5 ở trên và trạng thái trong `docs/experiments.md` trong cùng commit.
- Không commit file build LaTeX, dữ liệu lớn, checkpoint, khóa API (`.env`). Xem `.gitignore`.
