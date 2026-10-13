# CLAUDE.md — Hướng dẫn cho agent (Claude Code) làm tiếp báo cáo này

Đọc file này trước tiên, sau đó đọc `docs/context.md`. Không cần đọc lại hai bài báo từ đầu:
những gì đã kiểm chứng nằm trong `docs/findings.md`.

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
  findings.md           ← sự thật đã kiểm chứng: số liệu bài báo, điểm không khớp bài ↔ mã (có số dòng)
  experiments.md        ← danh sách thực nghiệm cần tái lập, trạng thái, chỗ dán link W&B
  plan_bao_cao.md       ← kế hoạch & sườn ban đầu (lịch sử; ưu tiên các file trên khi mâu thuẫn)
report/                 ← mã LaTeX của báo cáo
  main.tex              ← preamble, macro, thứ tự \input
  sections/00_cover.tex,
           A_intro.tex, A_background.tex, A_graphvis.tex, A_conclusion.tex (Phần A),
           B_intro.tex, B_related.tex, B_scimkg.tex, B_conclusion.tex (Phần B),
           04_contributions.tex, 99_appendix.tex
  figures/*.tex         ← hình TikZ (vẽ lại, không chụp hình bài báo)
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
| Cấu trúc báo cáo (12/10/2026) | Hai phần độc lập, mỗi phần có giới thiệu, kiến thức nền và kết luận riêng; không còn giới thiệu/kiến thức nền/kết luận chung. Đã bỏ tóm tắt và từ khóa chung; sau tiêu đề/tác giả vào thẳng Phần A. Phần B mở ở trang mới; mỗi phần đánh số mục lại từ I (Phần A I–IX, Phần B I–VIII). Đã biên dịch XeLaTeX (13/10/2026, sau khi viết Giới thiệu và Công trình liên quan của Phần B): **14 trang**, không lỗi, không tràn lề, không tham chiếu chưa giải quyết. |
| A: I Giới thiệu | Đã viết đoạn giới thiệu ngắn |
| B: I Giới thiệu (13/10/2026) | Bản nháp đầy đủ: bối cảnh, hai hạn chế (có trích dẫn KnowEdu, MOOCCubeX, Jhajj, TIVA-KG), ý tưởng và số liệu theo bài, đóng góp của báo cáo, lộ trình. Link bài và mã dạng chú thích. Lộ trình chưa có `\ref`: thêm khi tái cấu trúc Phần B thành 5 mục (Giới thiệu, Công trình liên quan, Phương pháp, Thực nghiệm, Kết luận). |
| A: II Kiến thức nền | **Sườn** |
| B: II Công trình liên quan (13/10/2026) | Bản nháp đầy đủ trong `B_related.tex` (thay `B_background.tex`): 5 mục con, mỗi mục một đoạn, kèm bảng so sánh đồ thị đa phương thức. Mô tả công trình lấy từ tóm tắt hoặc trang dự án, chưa đọc toàn văn. |
| A: III–VI, VIII (GraphVis) | Bản nháp hoàn chỉnh |
| A: VII Tái lập; VI-C thực nghiệm độ phức tạp ảnh | **Chờ chạy** (bảng `\todo`) |
| B: III–V, VII (SciMKG) | Bản nháp hoàn chỉnh, đã đối chiếu mã nguồn |
| B: VI Tái lập | **Chờ chạy** |
| A: IX Kết luận; B: VIII Kết luận; Phân công, Phụ lục B–C | **Sườn** |
| Code tái lập | GraphVis ở `repro/graphvis/`; SciMKG phát triển trong fork `SciMKG/`. |
| Chạy thực nghiệm trên Kaggle/Colab, dán link W&B | **Việc tiếp theo** — xem `docs/experiments.md` |

Việc tiếp theo: notebook GraphVis; sửa và kiểm thử SciMKG trong fork.
Kết quả đã có từ local: `score-authors` (đầu ra gốc của tác giả, α=0,7) P .953 R .740 F1 .833 MCC .690 — xem `docs/findings.md` §B4.
Sai khác GraphVis: `repro/graphvis/DEVIATIONS.md`; SciMKG: ghi tại fork.

## 6. Làm việc nhóm (2 người, 2 agent)

- Mỗi người làm trên **nhánh riêng** (`khang/...`, `nhi/...`), mở PR vào `main`; `git pull --rebase` trước khi bắt đầu.
- Tránh sửa cùng một file cùng lúc. Phân chia gợi ý: một người giữ `sections/A_graphvis.tex` + `repro/graphvis/`,
  người kia giữ `sections/B_scimkg.tex` + fork `SciMKG/`. `main.tex`, `refs.bib` sửa nhỏ, báo trước.
- Khi xong một việc: cập nhật bảng §5 ở trên và trạng thái trong `docs/experiments.md` trong cùng commit.
- Không commit file build LaTeX, dữ liệu lớn, checkpoint, khóa API (`.env`). Xem `.gitignore`.
