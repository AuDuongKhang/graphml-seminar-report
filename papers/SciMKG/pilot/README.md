# Chạy thử SciMKG

EVIA gọi trực tiếp các hàm trong fork `SciMKG/ExtractMultimodalConcepts/NER.py`, qua `main_evia.py`. Pilot chuyển phản hồi Codex JSON sang danh sách Markdown và điều chỉnh ngôn ngữ đầu ra; dùng một mô hình, khác cấu hình ba LLM của bài. Không dùng mã `repro/` của PR #1 trong kho báo cáo. Mở [pilot-report.html](report/pilot-report.html) để xem báo cáo duy nhất.

```text
pilot/
  report/
    pilot-report.html Báo cáo duy nhất, mở trực tiếp trong trình duyệt
  README.md           Hướng dẫn chạy
  *.py                Bộ chạy pilot, bộ chuyển EVIA và kiểm tra
  requirements*.txt   Các phụ thuộc
  data/               Đầu vào, nhãn riêng và nguồn đã chụp (cục bộ)
  runs/               Đồ thị, âm thanh, cache và nhật ký (cục bộ)
  upstream/           Mã gốc và PDF để đối chiếu (cục bộ)
  results/            Bản ghi kiểm tra, nguồn dữ liệu và mã băm
```

## Môi trường

- Python ≥ 3.11; `pytest` để kiểm thử.
- Submodule `SciMKG/` đã khởi tạo; pilot dùng phiên bản fork hiện có trên máy.
- `beautifulsoup4` để chụp nội dung Wikipedia; `ffmpeg` để đổi âm thanh thành WAV, `ffprobe` để kiểm tra clip video.
- Codex CLI có `exec --output-schema`, đăng nhập ChatGPT. Không đưa khóa API vào lệnh.
- Tùy chọn trên macOS: `say` với giọng `Linh` cho tiếng Việt hoặc `Tingting` cho tiếng Trung. Cần người nghe đánh giá.
- Không cần GPU để chạy các kiểm tra cấu trúc.

```bash
cd papers/SciMKG/pilot
python3 -m pip install -r requirements.txt
python3 -m pytest -q test_pilot.py
codex login status
```

## Chuẩn bị chỉ một mẫu

Tải `Example.zip` của tác giả, kiểm tra mã băm và đọc danh sách thành phần trước khi chọn một bài. Nếu mẫu không có nhãn hoặc phiếu, chỉ lấy thành phần thuộc bài đó từ bản công bố; không chạy trên toàn bộ tập. Dữ liệu tải về và lượt chạy nằm ở `data/`, `runs/`, không đưa vào git.

```bash
python3 prepare_sample.py --download --extract-to data/example
```

Nếu đã tải tệp qua trình duyệt, dùng `--archive đường/dẫn/Example.zip --extract-to data/example`. Script chỉ chấp nhận mã băm của tệp mẫu đã công bố; không tải `RawData.zip`. Lưu JSON danh mục mà lệnh in ra cùng dữ liệu để truy vết nguồn. Nếu gói thiếu văn bản/nhãn, `remote_sample.py` lấy chỉ mục và thành phần chỉ của bài đã chọn:

```bash
python3 remote_sample.py RawData.zip --inventory data/raw-inventory.json
python3 remote_sample.py Groundtruth.zip --inventory data/gold-inventory.json
```

Đọc danh mục JSON rồi thêm `--member 'đường/dẫn/trong/zip'` cho từng thành phần và `--destination data/author`. Bộ đọc yêu cầu HTTP 206, không tải toàn bộ ZIP khi máy chủ bỏ qua Range. CRC và SHA-256 từng thành phần được lưu trong `download-receipt.json`; MD5 ZIP lớn chưa được kiểm tra. Chỉ mục và khoảng byte đã đọc có cache; tên tệp không gắn UTF-8 dùng GBK. Phạm vi đã chọn và hash tài nguyên nằm trong `results/data-provenance.json`.

Đổi đầu vào đã chọn sang một tệp JSON theo `lesson.template.json`. Với hai mẫu của trang duyệt hiện tại, dùng `id=author-mitochondria` và `id=vi-photosynthesis`; `render_report.py` nhận đúng hai tên thư mục này. Giữ video và ảnh cùng nguồn; trường `video_url` phải chỉ tới video, không dùng URL trang văn bản thay thế. Ảnh dùng đường dẫn tương đối với tệp JSON. Không thêm nhãn chuẩn vào tệp đầu vào mô hình. Nếu không xác minh được video, ghi rõ thiếu nguồn/lỗi phát; không lấy video của bài khác để lấp chỗ trống.

Với clip cục bộ, thêm `video_path` (tương đối với tệp JSON) và `video_clip_start_seconds` (mốc bắt đầu trong video nguồn). Mốc `segments` vẫn theo video nguồn; bộ chạy kiểm tra mọi đoạn nằm trong clip và lưu thêm mốc cục bộ. Có thể chuẩn bị lại clip đã chọn bằng yt-dlp trong môi trường riêng:

```bash
.venv/bin/python -m pip install -r requirements-video.txt
.venv/bin/yt-dlp --js-runtimes node -f 'bv*[height<=360][vcodec^=avc1]+ba[acodec^=mp4a]/b[height<=360][ext=mp4]' --download-sections '*00:01:14-00:02:03' --force-keyframes-at-cuts --merge-output-format mp4 -o 'data/vi-photosynthesis-74-123.mp4' 'https://www.youtube.com/watch?v=_nmB096nlcM'
```

Lệnh cần Node và ffmpeg. Giữ nguồn/biên/mã băm trong bản ghi provenance; `video_path` không thay thế `video_url`. Nếu chưa có `.venv`, tạo theo phần W&B bên dưới trước khi cài tùy chọn video.

```bash
python3 pilot.py check-input data/author-lesson.json
python3 pilot.py check-author-votes 'đường/dẫn/tệp_Validationner'
python3 pilot.py run data/author-lesson.json --gold data/author-gold.json --use-conceptnet --candidates data/author-conceptnet.json --use-wikipedia --evidence data/author-sources/wikipedia.json --speech
```

### Nhãn đánh giá riêng

Tệp `--gold` có dạng `{"status":"frozen","concepts":["..."],"provenance":"nhãn công bố / hai người đánh giá, ngày ..."}`. Chỉ dùng khi nhãn tương ứng đúng mẫu. Nhãn được đọc sau các lời gọi mô hình.

### Nguồn ngoài riêng

- Hai cờ `--use-conceptnet` và `--use-wikipedia` mặc định tắt, giống mã chính. Chỉ truyền đường dẫn không bật nguồn ngoài.
- `--use-conceptnet --candidates`: bản ghi `source_concept`, `concept`, `relation`, `url` từ ConceptNet. Bộ luật trích danh từ trong mã chính còn trống, nên hiện không thêm khái niệm. Không giữ cách bổ sung riêng của lượt pilot cũ.
- `--use-wikipedia --evidence`: bản ghi `concept`, `url`, `text` chụp từ Wikipedia. Giữ URL, văn bản và thời điểm thu thập. Khi thiếu nguồn, mã chính vẫn yêu cầu LLM viết giải thích; pilot ghi `grounded=false` và `llm_fallback_unverified`.
- Bật ít nhất một nguồn sẽ tạo giải thích theo mã chính. Khi cả hai tắt, không tạo giải thích hoặc âm thanh từ giải thích.
- Bộ chạy nhận ảnh chụp dữ liệu nguồn ngoài, không tự xác minh nội dung URL. Thu thập và kiểm chứng nguồn là bước chuẩn bị bắt buộc.

`collect_sources.py` nhận ánh xạ JSON từ tên khái niệm dự đoán sang tên bài Wikipedia đã xác định; lưu bản HTML, đoạn nguồn, thời điểm, hash và lỗi. Dùng ví dụ:

```bash
python3 collect_sources.py data/author-wiki-mapping.json --language zh --out data/author-sources
python3 collect_sources.py data/vi-wiki-mapping.json --language vi --out data/vi-sources
```

Nếu API ConceptNet lỗi, có thể lấy cạnh từ trang tra cứu công khai. Phải lưu URL cạnh, nhãn gốc và nguồn của bản dịch; không thêm ứng viên theo trí nhớ rồi ghi là ConceptNet.

## Bài tiếng Việt sau mẫu tác giả

Sau khi mẫu tác giả chạy và xuất đồ thị hợp lệ, dùng tệp `summary.json` của nó làm điểm kiểm tra thứ tự. Điểm kiểm tra này chỉ chứng minh mã đã chạy trước trên mẫu nguồn; không chứng minh chất lượng đã được người đánh giá duyệt.

```bash
python3 pilot.py run data/vietnamese-lesson.json --author-checkpoint runs/main-evia/author-mitochondria/summary.json --gold data/vietnamese-gold.json --use-conceptnet --candidates data/vietnamese-conceptnet.json --use-wikipedia --evidence data/vi-sources/wikipedia.json --speech
```

`--model` mặc định là `gpt-6.1-sol`; tên này phải có quyền truy cập trong tài khoản. Nếu không có, bộ chạy dừng; không tự đổi mô hình hoặc gọi API trả phí. `--rounds` mặc định 5, `--alpha` mặc định 0,7 và nhận giá trị trong [0,1]. Trích xuất giữ khái niệm ở lần xuất hiện đầu tiên, giống mã chính. Phải hoàn tất mọi vòng kiểm chứng; lỗi dừng lượt, không giảm mẫu số bỏ phiếu.

Lượt mới mặc định lưu ở `runs/main-evia/`, cache tại `runs/main-evia/cache/`. Các lượt lịch sử ở `runs/author-mitochondria/` và `runs/vi-photosynthesis/` giữ nguyên. Nếu mã hoặc cấu hình thay đổi, chọn `--out` mới; bộ chạy từ chối ghi đè lượt khác cấu hình. Nhật ký lưu prompt, phản hồi JSON, thông tin sử dụng và thời gian; giữ cục bộ.

Đối chiếu EVIA bằng phản hồi đã lưu, không gọi Codex hoặc API:

```bash
python3 audit_run.py --check-main-evia
```

Kết quả ở `results/main-evia-check.json`: mẫu tác giả trước, tiếng Việt sau; kiểm tra bỏ phiếu, cờ nguồn, giải thích thiếu nguồn và xuất JSON/RDF. Đây là kiểm tra hành vi mã, chưa phải dự đoán mới với prompt mới. API Wikipedia trực tiếp và cấu hình dịch/proxy của mã chính chưa được kiểm chứng. Định dạng cache văn bản cũ của mã chính tách từ theo khoảng trắng khi đọc lại; pilot dùng tệp tạm mới và JSON để giữ nguyên thuật ngữ.

Sau khi chạy, đối chiếu độc lập rồi cập nhật phần duyệt trong báo cáo:

```bash
python3 audit_run.py runs/main-evia/author-mitochondria --cache runs/main-evia/cache
python3 audit_run.py runs/main-evia/vi-photosynthesis --cache runs/main-evia/cache
```

Báo cáo hiện nhúng đầu ra lịch sử cùng mục đồng bộ EVIA; liên kết tới tệp đầy đủ yêu cầu giữ thư mục `runs/`. Nhận xét lưu cục bộ trên trình duyệt và có nút xuất JSON. `render_report.py` cập nhật hai mẫu lịch sử và mục đồng bộ trong cùng HTML, không gọi mô hình. Đầu ra của lượt mới được audit riêng; chưa thay thế phần duyệt lịch sử.

## Kiểm tra thủ công và công khai kết quả

Hai người chấm độc lập liên kết ảnh/video và lời giải thích; nghe từng tệp âm thanh. Không gọi lượt chạy là hoàn tất chất lượng khi các trường đánh giá vẫn là `pending`.

Mẫu tác giả có bản dịch hỗ trợ đọc trong `results/author-vietnamese-view.json`; giữ bản gốc tiếng Trung để đối chiếu. Bản dịch gắn hash với nguồn; nếu nguồn đổi, cập nhật bản dịch trước khi sinh lại báo cáo. Người không biết tiếng Trung có thể ghi nhận xét nội dung/ảnh qua bản dịch, còn chấm phát âm và kiểm chứng ngôn ngữ gốc cần người biết tiếng Trung. Chọn “Chưa đủ dữ liệu” cho phần chưa thể đánh giá. JSON nhận xét ghi rõ căn cứ là bản dịch hay tiếng Việt gốc.

Ghi W&B ngoại tuyến trong môi trường riêng:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-wandb.txt
.venv/bin/python log_wandb.py --run runs/author-mitochondria --run runs/vi-photosynthesis --directory runs/wandb-offline
```

Chỉ sau khi đăng nhập và xác định entity/dự án **công khai**, dùng `--online --entity ten-entity --project ten-project`, hoặc đồng bộ đúng thư mục `offline-run-*` đã tạo bằng W&B CLI. Kiểm tra đường dẫn xem công khai được rồi ghi vào báo cáo. Dự án ngoại tuyến chưa tạo URL công khai; không tự coi đó là công bố kết quả.

`log_wandb.py` chỉ ghi cấu hình và chỉ số, đánh dấu nhãn tạm/chờ người duyệt; không đăng prompt, dữ liệu gốc hay sự kiện đăng nhập. `--online` không tự đổi quyền riêng tư của dự án. Không chạy lại lệnh ngoại tuyến nếu chỉ muốn xem lượt đã lưu.
