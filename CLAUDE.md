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
README.md               ← hướng dẫn CHẠY tái lập (yêu cầu môn học; KHÔNG ghi kết quả/giải pháp ở đây)
docs/
  context.md            ← yêu cầu môn học, nhận xét của GV, quyết định đã chốt
  findings.md           ← sự thật đã kiểm chứng: số liệu bài báo, điểm không khớp bài ↔ mã (có số dòng)
  experiments.md        ← danh sách thực nghiệm cần tái lập, trạng thái, chỗ dán link W&B
  plan_bao_cao.md       ← kế hoạch & sườn ban đầu (lịch sử; ưu tiên các file trên khi mâu thuẫn)
report/                 ← mã LaTeX của báo cáo
  main.tex              ← preamble, macro, thứ tự \input
  sections/00_cover.tex, 01_intro.tex, 02_background.tex,
           A_graphvis.tex (Phần A), B_scimkg.tex (Phần B),
           03_conclusion.tex, 99_appendix.tex
  figures/*.tex         ← hình TikZ (vẽ lại, không chụp hình bài báo)
  refs.bib
GraphVis-code/          ← mã gốc GraphVis (Apache-2.0). Commit đầu tiên = bản gốc chưa sửa;
                          mọi sửa đổi để tái lập xem bằng `git diff <commit đầu> -- GraphVis-code`
SciMKG/                 ← git submodule trỏ tới kg-bnu/SciMKG (chỉ đọc, không sửa trong submodule)
repro/                  ← (sẽ tạo) code tái lập của nhóm: notebook Colab, script, cấu hình W&B
```

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
| I Giới thiệu, II Kiến thức nền | **Sườn** (ý chính nằm trong comment `%` của file) |
| A: III–VI, VIII (GraphVis) | Bản nháp hoàn chỉnh |
| A: VII Tái lập; VI-C thực nghiệm độ phức tạp ảnh | **Chờ chạy** (bảng `\todo`) |
| B: IX–XI, XIII (SciMKG) | Bản nháp hoàn chỉnh, đã đối chiếu mã nguồn |
| B: XII Tái lập | **Chờ chạy** |
| Kết luận, Phân công, Tóm tắt, Phụ lục B–C | **Sườn** |
| Repo code tái lập (`repro/`) + README chạy | **Chưa làm** — việc tiếp theo |

Việc tiếp theo được đề xuất (theo thứ tự): xem `docs/experiments.md`.

## 6. Làm việc nhóm (2 người, 2 agent)

- Mỗi người làm trên **nhánh riêng** (`khang/...`, `nhi/...`), mở PR vào `main`; `git pull --rebase` trước khi bắt đầu.
- Tránh sửa cùng một file cùng lúc. Phân chia gợi ý: một người giữ `sections/A_graphvis.tex` + `repro/graphvis/`,
  người kia giữ `sections/B_scimkg.tex` + `repro/scimkg/`. `main.tex`, `refs.bib` sửa nhỏ, báo trước.
- Khi xong một việc: cập nhật bảng §5 ở trên và trạng thái trong `docs/experiments.md` trong cùng commit.
- Không commit file build LaTeX, dữ liệu lớn, checkpoint, khóa API (`.env`). Xem `.gitignore`.
