"""Cập nhật phần duyệt và bằng chứng trong báo cáo hợp nhất; không gọi mô hình."""
import base64
import html
import json
from pathlib import Path

from pilot import digest, write_json

ROOT = Path(__file__).resolve().parent
IMAGE_LABELS = {"author-plant-cell":"Tế bào thực vật", "author-chloroplast-negative":"Lục lạp, ảnh đối chứng", "vi-photosynthesis-diagram":"Quang hợp, hình cùng Bài 22"}
AUTHOR_VIEW_PATH = ROOT / "results" / "author-vietnamese-view.json"
AUTHOR_COMPARISON_PATH = ROOT / "results" / "author-output-comparison.json"
TRANSLATIONS = json.loads(AUTHOR_VIEW_PATH.read_text(encoding="utf-8"))["terms"]


def read(path): return json.loads(path.read_text(encoding="utf-8"))
def esc(value): return html.escape(str(value), quote=True)
def name(term): return TRANSLATIONS.get(term, term)
def data_uri(path, mime): return "data:" + mime + ";base64," + base64.b64encode(path.read_bytes()).decode()


def author_output_comparison():
    reference = read(AUTHOR_COMPARISON_PATH)
    for filename, expected in reference["source_files"].items():
        if digest((ROOT / filename).read_bytes()) != expected:
            raise ValueError(f"Đối chiếu tác giả cần cập nhật vì nguồn đã đổi: {filename}")
    rows = "".join(f'<tr><td>{esc(x["stage"])}</td><td>{esc(x["author_output"])}</td><td>{esc(x["checked"])}</td><td>{esc(x["limit"])}</td></tr>' for x in reference["comparison_rows"])
    terms = {**TRANSLATIONS, "结构模式图": "sơ đồ cấu trúc"}
    score_rows = "".join(f'<tr><td>{esc(x["interval"])}</td><td>{esc(terms[x["concept"]])}</td><td>{x["raw_count"]}</td><td>{x["normalized_score"]:.3f}</td><td>{"Giữ" if x["passes_threshold"] else "Loại"}</td></tr>' for x in reference["votes"]["rows"])
    sources = " · ".join(f'<a href="../{esc(path)}">{label}</a>' for suffix,label in [("Extractionner","Trích xuất gốc"),("Validationner","Phiếu gốc"),("Intergrationner","Tích hợp gốc")] for path in reference["source_files"] if path.endswith(suffix))
    example = reference["example"]
    image = data_uri(ROOT / example["folder"] / "image.jpg", "image/jpeg")
    return f'''<details id="author-output-comparison" class="author-evidence" open><summary>Đối chiếu với đầu ra tác giả · bằng chứng sẵn có</summary>
<p><b>Khớp tên 5/6 nhãn tạm:</b> mục chưa khớp là “hoạt động sống”; kết quả có “hoạt động sống của tế bào”. Có thể gần nghĩa trong ngữ cảnh, nhưng chưa được phân xử. 12 khái niệm còn lại nằm ngoài sáu nhãn tạm; chưa đủ căn cứ kết luận tất cả đều đúng hoặc sai.</p>
<div class="table-wrap"><table><thead><tr><th>Bước</th><th>Tác giả đã cung cấp gì?</th><th>Đã kiểm tra được gì?</th><th>Giới hạn kết luận</th></tr></thead><tbody>{rows}</tbody></table></div>
<details><summary>Xem phép tính phiếu và tệp tích hợp của tác giả</summary><p>Chuẩn hóa (s−1)/15; giữ điểm &gt; 0,7. Đây là phép tính lại từ phiếu công bố, không phải lượt suy luận mới.</p><div class="table-wrap"><table><thead><tr><th>Đoạn</th><th>Khái niệm qua bản dịch</th><th>Phiếu gốc s</th><th>Điểm tính lại</th><th>Qua ngưỡng</th></tr></thead><tbody>{score_rows}</tbody></table></div><p class="notice">{esc(reference["votes"]["interpretation"])}</p><p>{sources}</p></details>
<details><summary>Xem mẫu tham chiếu ti thể của tác giả</summary><p>{esc(example["text_vi"])}</p><figure><img style="max-width:221px" src="{image}" alt="Ảnh ti thể được tác giả gắn trong mẫu Mitochondria"><figcaption>Ảnh của mẫu Example.zip; chưa có nhãn chuẩn liên kết cho ảnh đầu vào pilot.</figcaption></figure><p><a href="../data/example/Mitochondria/text.txt">Lời giải thích gốc tiếng Anh</a> · <a href="../data/example/Mitochondria/Mitochondria.mp3">Âm thanh mẫu của tác giả</a> · <a href="../data/example/Mitochondria/video.txt">URL và mốc video mẫu</a></p><p class="small">Bản diễn đạt tiếng Việt do Codex tạo. Mốc video mẫu 01:22–02:12 chưa được xác nhận cùng bài với trích đoạn pilot 06:35–08:07. Mẫu này hỗ trợ đối chiếu tài nguyên, chưa phải bộ đáp án chuyên gia.</p></details>
<p class="notice">Bạn không cần tự chấm kiến thức sinh học hoặc tiếng Trung để hoàn tất phần kiểm tra kỹ thuật. Có thể nhận xét phạm vi thăm dò; phần chất lượng chưa kiểm chứng ghi “Chưa đủ dữ liệu”. Đồng ý phạm vi nhãn không xác nhận mọi dự đoán hay mọi liên kết đều đúng.</p><p class="small"><a href="../results/author-output-comparison.json">Nguồn, mã băm và kết quả đối chiếu</a></p></details>'''


def sample(folder, title, concept, lesson_file):
    run = ROOT / "runs" / folder
    summary, audit, graph = read(run / "summary.json"), read(run / "audit.json"), read(run / "graph.json")
    lesson, input_hash = read(run / "input.json"), digest((run / "input.json").read_bytes())
    extracted, additions, image_links = read(run / "extraction.json"), read(run / "augmentation.json"), read(run / "image-links.json")
    terms = list(dict.fromkeys(t for part in extracted for t in part["concepts"]))
    explanations = read(run / "explanations.json")
    explanation = explanations[concept]
    is_author = folder == "author-mitochondria"
    view = read(AUTHOR_VIEW_PATH) if is_author else None
    if view:
        for filename, expected in view["source_files"].items():
            if digest((ROOT / filename).read_bytes()) != expected:
                raise ValueError(f"Bản dịch cần cập nhật vì nguồn đã đổi: {filename}")
        if set(view["explanations"]) != set(explanations) or set(view["segments"]) != {s["id"] for s in lesson["segments"]}:
            raise ValueError("Bản dịch không bao phủ đúng mẫu tác giả")
    display_text = view["explanations"][concept] if view else explanation["text"]
    audio_node = next(n for n in graph["nodes"] if n["kind"] == "audio" and n["id"] in [b for a,r,b in graph["edges"] if r == "hasAudio" and a == next(n["id"] for n in graph["nodes"] if n["kind"] == "concept" and n["label"] == concept)])
    audio_options = []
    for term, info in explanations.items():
        cid = next(n["id"] for n in graph["nodes"] if n["kind"] == "concept" and n["label"] == term)
        aid = next(b for a,r,b in graph["edges"] if a == cid and r == "hasAudio")
        node = next(n for n in graph["nodes"] if n["id"] == aid)
        resource = str(Path("..") / Path(node["path"]).relative_to(ROOT))
        src = "#embedded" if term == concept else resource
        text = view["explanations"][term] if view else info["text"]
        suffix = " (bổ sung)" if term in additions["concepts"] else ""
        audio_options.append(f'<option value="{esc(src)}" data-explanation="{esc(text)}" data-original-explanation="{esc(info["text"])}" data-source-urls="{esc(json.dumps(info["source_urls"]))}" {"selected" if term == concept else ""}>{esc(name(term) + suffix)}</option>')
    imgs = []
    for image in lesson["images"]:
        path = Path(image["path"])
        if not path.is_absolute(): path = lesson_file.parent / path
        imgs.append(f'<figure><img src="{data_uri(path,"image/png")}" alt="Hình nguồn của {esc(title)}"><figcaption>{esc(IMAGE_LABELS[image["id"]])}</figcaption></figure>')
    matches = "".join(f'<tr><td>{esc(IMAGE_LABELS[m["image_id"]])}</td><td>{esc(name(m["concept"]))}</td><td>{m["confidence"]:.2f}</td></tr>' for m in image_links)
    provisional = summary["extraction_metrics"]
    usage = audit["usage"]
    chips = "".join(f'<li>{esc(name(t))}</li>' for t in terms)
    labels = [("labels","Phạm vi nhãn"),("images","Liên kết ảnh"),("video","Đoạn video"),("explanation","Lời giải thích"),("audio","Phát âm và nội dung âm thanh")]
    if view:
        labels = [("labels","Phạm vi sáu nhãn tạm qua bản dịch"),("images","Liên kết ảnh theo khái niệm đã dịch"),("video","Video tác giả · thiếu nguồn"),("explanation","Nội dung lời giải thích qua bản dịch"),("audio","Phát âm tiếng Trung · cần người biết ngôn ngữ")]
    controls = "".join(f'<label for="review-{folder}-{key}">{label}<select id="review-{folder}-{key}" name="{folder}-{key}" data-review="{folder}:{key}"><option value="pending">Chưa duyệt</option><option value="accepted">{"Phạm vi phù hợp để thăm dò" if view and key == "labels" else "Đạt theo bản dịch" if view and key in {"images","explanation"} else "Đạt"}</option><option value="needs_changes">Cần sửa</option><option value="unavailable">Chưa đủ dữ liệu</option></select></label>' for key,label in labels)
    translated_lesson = original_details = gold_review = review_help = ""
    reading_label = "Chọn khái niệm để nghe"
    explanation_heading = "Lời giải thích và giọng nói"
    audio_help = "Chọn được mọi âm thanh của mẫu; ngoài hai âm thanh nhúng, cần giữ thư mục lượt chạy. Bạn cần nghe phát âm và xem lại nội dung."
    if view:
        reading_label = "Chọn khái niệm để đọc bản dịch"
        explanation_heading = "Lời giải thích bằng tiếng Việt"
        translated_segments = "".join(f'<h4>{esc(s["start"])}–{esc(s["end"])}</h4><p lang="vi">{esc(view["segments"][s["id"]])}</p>' for s in lesson["segments"])
        notes = " ".join(x["note_vi"] for x in view["source_notes"])
        translated_lesson = f'<details class="translated-lesson" open><summary>Đọc hai trích đoạn bài giảng bằng tiếng Việt</summary>{translated_segments}<p class="small">{esc(notes)}</p></details>'
        original_segments = "".join(f'<h4>{esc(s["start"])}–{esc(s["end"])}</h4><p lang="zh">{esc(s["text"])}</p>' for s in lesson["segments"])
        mappings = "".join(f'<tr><td>{esc(name(t))}</td><td lang="zh">{esc(t)}</td></tr>' for t in view["terms"])
        original_details = f'<details class="original-language"><summary>Bản gốc tiếng Trung để đối chiếu</summary><h4>Lời giải thích đang chọn</h4><p class="source-explanation" lang="zh">{esc(explanation["text"])}</p><h4>Bản chép lời gốc</h4>{original_segments}<div class="table-wrap"><table><thead><tr><th>Nghĩa tiếng Việt</th><th>Nhãn gốc</th></tr></thead><tbody>{mappings}</tbody></table></div></details>'
        gold = read(ROOT / "data" / "author-gold.json")
        gold_rows = "".join(f'<tr><td>{esc(name(t))}</td><td>{"Khớp tên gốc" if t in terms else "Chưa khớp tên gốc"}</td></tr>' for t in gold["concepts"])
        outside = ", ".join(name(t) for t in provisional["false_positive"])
        gold_review = f'<h4>Sáu nhãn tạm của trích đoạn, đọc bằng tiếng Việt</h4><div class="table-wrap"><table><thead><tr><th>Nhãn tạm</th><th>Đối chiếu tên trong kết quả gốc</th></tr></thead><tbody>{gold_rows}</tbody></table></div><p>“Hoạt động sống” và “hoạt động sống của tế bào” chưa được tự gộp. Nhãn ngoài phạm vi tạm gồm: {esc(outside)}. Chúng chưa được kết luận là dự đoán sai.</p>'
        review_help = '<p class="notice">Mục “Phạm vi sáu nhãn tạm” chỉ ghi nhận phạm vi phù hợp để thăm dò; khớp 5/6 tên chưa xác nhận mọi dự đoán đúng. Bạn không cần tự chấm phần kiến thức sinh học hoặc tiếng Trung chưa thể xác minh: chọn “Chưa đủ dữ liệu”. Bảng đối chiếu với tác giả phía trên ghi rõ bằng chứng kỹ thuật đã có. Nhận xét qua bản dịch chưa xác nhận chất lượng bản gốc.</p>'
        audio_help = 'Âm thanh dưới đây vẫn đọc bản gốc tiếng Trung. Bạn có thể kiểm tra tệp phát được; việc chấm phát âm và khớp lời đọc cần người biết tiếng Trung. Bản dịch hỗ trợ đọc, chưa thay thế kiểm chứng ngôn ngữ gốc.'
    sources = " · ".join(f'<a href="{esc(url)}" target="_blank" rel="noreferrer">Wikipedia</a>' for url in explanation["source_urls"])
    video = f'<a href="{esc(lesson["video_url"])}" target="_blank" rel="noreferrer">Mở video nguồn</a>' if lesson.get("video_url") else 'Chưa có URL video đúng bài'
    if lesson.get("video_path"):
        path = Path(lesson["video_path"])
        if not path.is_absolute(): path = lesson_file.parent / path
        video = f'<video controls preload="metadata" aria-label="Video bài Việt, đoạn 01:14 đến 02:03" src="{data_uri(path,"video/mp4")}"></video><p class="small">Clip 49 giây, mốc gốc 01:14–02:03; hai đoạn trong clip: 0–35 và 35–49 giây.</p>' + video
    warning = 'Có bản dịch tiếng Việt để bạn đọc. Video nguồn còn thiếu; liên kết ti thể với ảnh tế bào thực vật ở điểm 0,99 cần đối chiếu hình và nội dung bài giảng.' if view else 'Đã có clip 49 giây và đã kiểm tra ánh xạ mốc thời gian. Hình lấy từ lý thuyết cùng Bài 22. Bạn cần duyệt nội dung và biên ngữ nghĩa.'
    trace = f'<div class="trace"><div><b>Văn bản</b><span>{esc(lesson["segments"][0]["start"])}–{esc(lesson["segments"][-1]["end"])}</span></div><div><b>{esc(name(concept))}</b><span>Khái niệm dự đoán</span></div><div><b>Ảnh</b><span>{sum(m["concept"]==concept for m in image_links)} liên kết, chờ duyệt</span></div><div><b>Video</b><span>{"Clip 49 giây, chờ duyệt" if lesson.get("video_path") else "Thiếu nguồn"}</span></div><div><b>Âm thanh</b><span>{audio_node["duration_seconds"]:.1f} giây, chờ nghe</span></div></div>'
    section = f'''<section id="{folder}" class="sample" aria-labelledby="heading-{folder}">
<h2 id="heading-{folder}">{esc(title)}</h2><p>{esc(lesson['sampling']['interval'])} · hai đoạn ngắn, không phải toàn bài. <a href="{esc(lesson['source_url'])}">Nguồn bài học</a></p>
<div class="stats"><div><b>{len(terms)}</b><span>khái niệm trích xuất</span></div><div><b>{len(additions['concepts'])}</b><span>khái niệm bổ sung</span></div><div><b>{summary['graph']['nodes']} / {summary['graph']['edges']}</b><span>nút / cạnh có hướng</span></div><div><b>{usage['actual_calls']}</b><span>lời gọi suy luận thật</span></div></div>
<p class="notice">{esc(warning)}</p>{trace}{author_output_comparison() if is_author else ''}{translated_lesson}
<div class="columns"><div><h3>Hình đưa vào mô hình</h3><div class="images">{''.join(imgs)}</div></div><div><h3>{explanation_heading}</h3><label>{reading_label} <select class="audio-select" id="audio-{folder}" name="audio-{folder}">{''.join(audio_options)}</select></label><p class="original" lang="vi">{esc(display_text)}</p>{original_details}<p class="small"><span class="explanation-sources">{sources}</span> · nguồn của lời giải thích đang chọn; mọi nguồn ở tệp lời giải thích</p><audio controls preload="metadata" src="{data_uri(Path(audio_node['path']),'audio/wav')}"></audio><div class="video-resource">{video}</div><p class="small">{audio_help}</p></div></div>
<details><summary>Khái niệm, liên kết ảnh và chỉ số tạm</summary><ul class="terms">{chips}</ul><div class="table-wrap"><table><thead><tr><th>Ảnh</th><th>Khái niệm</th><th>Điểm mô hình</th></tr></thead><tbody>{matches}</tbody></table></div><p><b>So khớp tên với nhãn tạm:</b> P = {provisional['precision']:.3f}; R = {provisional['recall']:.3f}; F1 = {provisional['f1']:.3f}; {provisional['tp']}/{provisional['gold']} nhãn được tìm thấy.</p><p class="small">{esc(provisional['annotation_provenance'])} Tên tương đương chưa tự gộp. Khái niệm ngoài nhãn tạm chưa chắc là dự đoán sai.</p>{gold_review}</details>
<details><summary>Thông tin sử dụng, kiểm tra cấu trúc và tệp đầy đủ</summary><p>{usage['input_tokens']:,} token đầu vào; {usage['output_tokens']:,} token đầu ra; {usage['cached_input_tokens']:,} token đầu vào được dịch vụ ghi nhận đã lưu đệm; {usage['elapsed_inference_seconds']:.2f} giây tổng thời gian lời gọi thật.</p><p>Đầu vào khớp mã băm; phiếu tính độc lập khớp; nút, nhãn và cạnh JSON/RDF khớp. Chưa chấm chất lượng bằng hai người. Tiền API chưa đo; xác thực ChatGPT.</p><p class="small">Mã băm đầu vào: <code>{input_hash}</code></p><p><a href="../runs/{folder}/graph.json">Đồ thị JSON</a> · <a href="../runs/{folder}/graph.rdf">RDF</a> · <a href="../runs/{folder}/explanations.json">Mọi lời giải thích</a> · <a href="../runs/{folder}/audit.json">Đối chiếu và thông tin sử dụng</a> · <a href="../runs/{folder}/summary.json">Tóm tắt lượt</a></p></details>
<details><summary>Ghi đánh giá của bạn cho mẫu này</summary>{review_help}<div class="review-form">{controls}</div><label class="comment-label" for="comment-{folder}">Nhận xét<textarea id="comment-{folder}" name="comment-{folder}" data-review="{folder}:comment" rows="3" placeholder="Ghi tên khái niệm hoặc liên kết cần sửa…"></textarea></label></details></section>'''
    return section, {"lesson_id": folder, "input_sha256": input_hash, "summary": summary, "usage": usage,
                     "checks": {k:audit[k] for k in ["input_snapshot_equal","rdf_nodes_equal","rdf_labels_equal","rdf_edges_equal","gold_file_markers_absent_in_prompts"]},
                     "files": {p.name:digest(p.read_bytes()) for p in [run/"input.json",run/"config.json",run/"graph.json",run/"graph.rdf",run/"audit.json",run/"summary.json"]}}


def replace_block(page, name, content):
    start, end = f"<!-- {name}_START -->", f"<!-- {name}_END -->"
    if page.count(start) != 1 or page.count(end) != 1 or page.index(start) >= page.index(end):
        raise ValueError(f"Báo cáo cần đúng một cặp mốc {name}")
    before, _, remainder = page.partition(start)
    _, _, after = remainder.partition(end)
    return before + start + "\n" + content + "\n" + end + after


def main_evia_check():
    from main_evia import fingerprint
    check = read(ROOT / "results/main-evia-check.json")
    current = fingerprint()
    if (check["source"]["sha256"] != current["sha256"] or
            check["source"]["adapter_sha256"] != current["adapter_sha256"] or
            check["runner_sha256"] != digest((ROOT / "pilot.py").read_bytes()) or
            check["audit_runner_sha256"] != digest((ROOT / "audit_run.py").read_bytes())):
        return '<section id="main-evia-check" class="report-panel"><h2>Đồng bộ EVIA trước tái cấu trúc · lịch sử</h2><p>Bằng chứng này thuộc phiên bản mã trước. Luồng mới đã dùng chung EVIA, căn chỉnh và lưu đồ thị; xem <a href="#source-reconstruction">kiểm tra mã chính hiện tại</a>. <a href="../results/main-evia-check.json">Giữ bản ghi cũ để truy vết</a>.</p></section>' 
    for item in check["samples"]:
        for path, expected in item["recorded_inputs_sha256"].items():
            if digest((ROOT / path).read_bytes()) != expected:
                raise ValueError("Đầu vào đối chiếu EVIA đã đổi: " + path)
    rows = "".join(f'<tr><td>{"Mẫu tác giả" if item["language"] == "zh" else "Bài tiếng Việt"}</td>'
                   f'<td>{item["segments"]} đoạn</td><td>{item["selected_unique_concepts"]} khái niệm</td>'
                   f'<td>{item["repeated_candidates_removed"]} lần lặp ở đoạn sau</td>'
                   '<td>Đạt cả 4 tổ hợp cờ</td></tr>' for item in check["samples"])
    return f'''<section id="main-evia-check" class="report-panel" aria-labelledby="main-evia-title">
<p class="eyebrow">Cập nhật {esc(check["date"])} · <a href="#e11">[E11]</a></p>
<h2 id="main-evia-title">EVIA hiện tại · dùng chung mã chính</h2>
<p><b>Pilot gọi trực tiếp EVIA của fork SciMKG</b> qua bộ chuyển <code>main_evia.py</code>. Đã chạy lại bằng phản hồi lưu sẵn: mẫu tác giả trước, bài tiếng Việt sau; <b>{check["actual_llm_calls"]} lời gọi LLM mới</b>. PR #1 không sử dụng là PR trong kho báo cáo.</p>
<ol class="evia-sync-flow" aria-label="Luồng EVIA dùng chung"><li><b>1 · Trích xuất</b><span>Giữ thuật ngữ đầy đủ; chỉ lần xuất hiện đầu.</span></li><li><b>2 · Kiểm chứng</b><span>Ứng viên gốc → phản hồi → tự sửa, 5 vòng.</span></li><li><b>3 · Tích hợp</b><span>Chuẩn hóa phiếu, giữ điểm ≥ 0,7.</span></li><li><b>4 · Bổ sung</b><span>ConceptNet / Wikipedia bật bằng cờ.</span></li></ol>
<details open><summary>Hành vi đã đồng bộ</summary><div class="table-wrap"><table><thead><tr><th>Bước</th><th>Mã chính và pilot mới</th><th>Điều đã kiểm tra</th></tr></thead><tbody>
<tr><td>Trích xuất</td><td>Khái niệm chỉ gắn với đoạn xuất hiện đầu; giữ cả cụm từ tiếng Việt.</td><td>Loại lặp giữa hai đoạn; “lục lạp”, “tế bào thực vật” không bị cắt từ.</td></tr>
<tr><td>Kiểm chứng</td><td>Giữ tập ứng viên gốc, tích lũy phản hồi; lỗi dừng lượt.</td><td>Lỗi trích xuất, phản hồi hoặc tự sửa không trở thành dự đoán rỗng; không chấp nhận vòng thiếu.</td></tr>
<tr><td>Tích hợp</td><td><code>(s − 1) / (M × k) ≥ α</code>; pilot M = 1, k = 5, α = 0,7.</td><td>Điểm và khái niệm khớp phép tính độc lập; biên 7/10 phiếu được giữ. Đây là công thức đã sửa của fork.</td></tr>
<tr><td>Bổ sung / giải thích</td><td>Hai cờ mặc định tắt. Bật nguồn mới tạo giải thích; thiếu Wikipedia thì dùng LLM.</td><td>Đã chạy đủ 4 tổ hợp cờ qua bộ chạy tới JSON/RDF. Giải thích thiếu nguồn ghi <code>grounded=false</code>.</td></tr>
</tbody></table></div></details>
<div class="table-wrap"><table><caption>Kết quả dùng phản hồi cũ với logic mã chính; chưa phải dự đoán mới</caption><thead><tr><th>Mẫu</th><th>Phạm vi</th><th>Qua tích hợp</th><th>Loại trùng</th><th>Nguồn ngoài</th></tr></thead><tbody>{rows}</tbody></table></div>
<p class="notice"><b>Giới hạn còn lại:</b> bộ luật ConceptNet trong mã chính còn trống, nên bật cờ hiện thêm <b>0 khái niệm</b>. API Wikipedia trực tiếp và cấu hình dịch/proxy chưa được kiểm chứng. Pilot vẫn dùng một mô hình Codex, chuyển JSON/ngôn ngữ và nguồn đã chụp; chưa trùng cấu hình LLM của bài. Kiểm tra ảnh dùng phản hồi giả lập rỗng, không tạo âm thanh mới.</p>
<p>Lượt chạy mới lưu riêng ở <code>runs/main-evia/</code>; mã hoặc cấu hình khác sẽ bị từ chối ghi đè lượt cũ. <b>Phần duyệt, đồ thị, âm thanh và số đo bên dưới thuộc lượt pilot trước khi đồng bộ</b>; giữ nguyên để truy vết.</p>
<details><summary>Phiên bản và bằng chứng kỹ thuật</summary><p>Fork: <code>{esc(check["source"]["revision"])}</code>; hậu tố <code>dirty</code> chỉ sửa đổi cục bộ chưa commit. SHA-256 của <code>NER.py</code>: <code>{esc(check["source"]["sha256"]["NER.py"])}</code>.</p><p>Chạy lại kiểm tra không gọi mô hình: <code>python audit_run.py --check-main-evia</code>. <a href="../results/main-evia-check.json">Bản ghi đối chiếu và mã băm</a> · <a href="../README.md">Cách chạy lượt mới</a>.</p><p>Cache văn bản cũ của mã chính còn tách cụm từ theo khoảng trắng khi đọc lại. Pilot gọi các hàm trên tệp tạm mới, rồi lưu JSON; chưa xác nhận khả năng tiếp tục lượt cũ của CLI chính với thuật ngữ nhiều từ.</p></details>
</section>'''



def alignment_review():
    baseline_path = ROOT / "results/alignment-author-review.json"
    if not baseline_path.exists():
        return ""
    path = ROOT / "results/alignment-author-review-v3.json"
    if not path.exists():
        path = ROOT / "results/alignment-author-review-v2.json"
        if not path.exists():
            path = baseline_path
    review = read(path)
    baseline = read(baseline_path)
    updated = path != baseline_path
    rows = ""
    for case in review["image_cases"]:
        if "author_label_vi" not in case:
            continue
        positive = "Khớp" if case["positive_matches_author"] else "Chưa khớp"
        swapped = "Từ chối" if case["rejects_swapped_label"] else "Giữ sai"
        before = next(row for row in baseline["image_cases"] if row["case_id"] == case["case_id"])
        previous = "Khớp" if before["positive_matches_author"] else "Chưa khớp"
        rows += f'<tr><td>{esc(case["author_label_vi"])}</td><td>{previous}</td><td>{positive}</td><td>{swapped}</td></tr>'
    findings = review.get("review_findings", [])
    notes = "".join(f'<p class="notice"><b>{esc(item["id"])} · {esc(item["status_vi"])}:</b> {esc(item["observation_vi"])} {esc(item["scope_vi"])}</p><p>{esc(item.get("followup_vi", ""))}</p><p><b>Tiếp theo:</b> {esc(item["next_step_vi"])}</p>' for item in findings)
    calls = (f'Trước sửa: {baseline["recorded_live_calls"]} lượt; sau sửa: {review["recorded_live_calls"]} lượt Codex mới.'
             if updated else f'Tổng {review["recorded_live_calls"]} lượt Codex mới.')
    if path.name.endswith("-v3.json"):
        calls += ' Lượt sửa đầu chưa khắc phục: 6 lượt; chẩn đoán đọc ảnh riêng: 1 lượt. Mỗi phiên bản dùng hai lời gọi mỗi ảnh.'
    displayed = review.get("presentation_vi", [
        {"case_id": case["case_id"], "concept_vi": name(link["concept"]),
         "location_vi": link.get("location", "Chưa ghi vị trí"),
         "visual_evidence_vi": link.get("visual_evidence", link["reason"])}
        for case in review["image_cases"] for link in case["links"]])
    locations = "".join(f'<p><b>{esc(item["case_id"])}</b>: {esc(item["concept_vi"])} · {esc(item["location_vi"])} · {esc(item["visual_evidence_vi"])}</p>' for item in displayed)
    return f'''<div id="alignment-review">
<h3>Duyệt căn chỉnh ảnh · 12/10/2026</h3>
<p>Chạy trực tiếp mã chính trên hai cặp trong <code>Example.zip</code>; thêm ảnh <code>12.png</code>. {calls} Phát lại cache không thêm lượt.</p>
<div class="table-wrap"><table><thead><tr><th>Nhãn tác giả</th><th>Trước sửa</th><th>Lượt hiện tại</th><th>Đổi chéo ti thể ↔ thận</th></tr></thead><tbody>{rows}</tbody></table></div>
<p class="small">Đổi chéo là đối chứng do nhóm tạo. Hai vai trò đề xuất–kiểm chứng dùng cùng Codex; chưa tái lập cấu hình mô hình của bài.</p>
{notes}
<details><summary>Xem vị trí và cấu trúc ở lượt hiện tại</summary>{locations}<p class="small">Diễn đạt tiếng Việt do Codex tạo để đọc; phản hồi gốc giữ trong bằng chứng.</p></details>
<p><a href="../results/alignment-author-review.json">A1 · trước sửa</a> · <a href="../results/{esc(path.name)}">A2 · lượt hiện tại</a></p>
</div>'''


def source_reconstruction():
    record = read(ROOT / "results/source-reconstruction.json")
    rows = ""
    for item in record["samples"]:
        folder = "../runs/main-source/" + item["lesson_id"]
        label = "Mẫu tác giả" if item["lesson_id"] == "author-mitochondria" else "Mẫu tiếng Việt"
        rows += f'<tr><td>{label} · 2 đoạn</td><td>{item["calls"]}</td><td>{item["nodes"]} / {item["edges"]}</td><td>{item["audio_files"]}</td><td><a href="{folder}/graph.json">JSON</a> · <a href="{folder}/graph.ttl">RDF</a> · <a href="{folder}/summary.json">Nhật ký lượt</a></td></tr>'
    tokens = sum(item["usage"].get("input_tokens", 0) for item in record["samples"])
    output_tokens = sum(item["usage"].get("output_tokens", 0) for item in record["samples"])
    image_module = "MultimodalAlignment/ImageAligment.py"
    recorded_hash = record["samples"][0]["source"]["sha256"][image_module]
    current_hash = digest((ROOT.parents[2] / "SciMKG" / image_module).read_bytes())
    snapshot_note = ('<p class="notice">Các lượt dựng đồ thị dưới đây thuộc mã trước khi sửa căn chỉnh ảnh. '
                     'Lượt kiểm tra sau sửa nằm ở <a href="#alignment-review">Duyệt căn chỉnh ảnh</a>; '
                     'chưa chạy lại toàn bộ bài sau thay đổi này.</p>' if current_hash != recorded_hash else "")
    return f'''<section id="source-reconstruction" class="report-panel">
<h2>Kiểm tra mã fork · Codex thuê bao</h2>
<p>Nhánh <code>nhi/scimkg-reconstruction</code>. Pilot gọi trực tiếp <code>SciMKG/pipeline.py</code> cho EVIA, bổ sung/giải thích, ảnh, TTS, JSON và RDF. Các kết quả cũ bên dưới được giữ làm lịch sử.</p>
{snapshot_note}
<div class="table-wrap"><table><thead><tr><th>Đầu vào</th><th>Lượt Codex mới</th><th>Nút / cạnh</th><th>WAV cục bộ</th><th>Đầu ra</th></tr></thead><tbody>{rows}</tbody></table></div>
<p>Hai mẫu dùng {record["new_inference_calls"]} lượt suy luận thuê bao ({tokens:,} token vào, {output_tokens:,} token ra; token gồm ngữ cảnh Codex); thêm một lượt kiểm tra kết nối. Phát lại mã tại phiên bản đã ghi nhận từ cache: <b>0 lượt mới</b>. Tiền API chưa đo.</p>
<p>Đã kiểm tra thuật ngữ nhiều từ, số phiếu/ngưỡng, cờ nguồn ngoài, ID/kiểu nút/quan hệ nghịch và JSON → RDF → JSON. Tiền xử lý đã kiểm tra SRT, PDF và video/âm thanh/clip cục bộ. Mỗi mẫu chỉ chạy thật một vòng kiểm chứng; năm vòng được kiểm tra bằng phản hồi kiểm soát.</p>
<p class="notice">Đây là kiểm tra mã và khả năng chạy, chưa phải Ex01–Ex02 toàn bài. Các vai trò LLM đều dùng cùng Codex; TTS dùng macOS. Mẫu tác giả thiếu video. Chưa chấm chất lượng chuyên môn; ASR cục bộ và dịch vụ API/nguồn ngoài trực tiếp chưa kiểm chứng.</p>
{alignment_review()}
<p><a href="../results/source-reconstruction.json">Nguồn, hash mã và đầu ra</a> · <a href="../../../../SciMKG/README.md">Cách chạy mã chính</a></p>
</section>'''

def main():
    first, e1 = sample("author-mitochondria", "1 · Mẫu nhỏ của tác giả: ti thể", "线粒体", ROOT / "data/author-lesson.json")
    second, e2 = sample("vi-photosynthesis", "2 · Bài tiếng Việt: quang hợp", "lục lạp", ROOT / "data/vietnamese-lesson.json")
    evidence = {"date":"2026-10-11", "status":"awaiting_human_review", "author_first":True, "pr1_code_used":False,
                "samples":[e1,e2], "api_cost_usd":None, "wandb_public_url":None,
                "limitations":["Nhãn tạm, chưa hai người chấm", "Mẫu tác giả thiếu video; clip Việt chờ duyệt nội dung", "Căn chỉnh ảnh cần xem lại", "W&B mới ngoại tuyến"]}
    evidence["presentation"] = {"display_language":"vi", "author_translation_file":"results/author-vietnamese-view.json",
                                "author_translation_sha256":digest(AUTHOR_VIEW_PATH.read_bytes()),
                                "translation_review":"pending", "original_language_review":"requires_Chinese_speaking_reviewer"}
    evidence["author_reference"] = {"file":"results/author-output-comparison.json", "sha256":digest(AUTHOR_COMPARISON_PATH.read_bytes()),
                                    "published_integration_matches_recomputed_votes":False, "expert_quality_review":"unassessed"}
    write_json(ROOT / "results/evidence.json", evidence)
    target = ROOT / "report" / "pilot-report.html"
    page = target.read_text(encoding="utf-8")
    page = replace_block(page, "PILOT_REVIEW", first + "\n" + second)
    minimal = {"date":evidence["date"], "presentation":evidence["presentation"], "author_reference":evidence["author_reference"],
               "review_criteria":{"author-mitochondria:labels":"Phạm vi phù hợp để thăm dò; không phải xác nhận mọi dự đoán đúng"},
               "review_basis":{"author-mitochondria":"Bản dịch tiếng Việt hỗ trợ đọc; chưa xác nhận tiếng Trung gốc",
                               "vi-photosynthesis":"Nội dung tiếng Việt gốc"},
               "samples":[{"lesson_id":x["lesson_id"],"input_sha256":x["input_sha256"]} for x in evidence["samples"]]}
    payload = json.dumps(minimal, ensure_ascii=False).replace("<", "\\u003c")
    page = replace_block(page, "PILOT_EVIDENCE", '<script id="pilot-evidence" type="application/json">' + payload + '</script>')
    if (ROOT / "results/main-evia-check.json").is_file():
        page = replace_block(page, "MAIN_EVIA_CHECK", main_evia_check())
    if (ROOT / "results/source-reconstruction.json").is_file():
        block = source_reconstruction()
        if '<!-- SOURCE_RECONSTRUCTION_START -->' in page:
            page = replace_block(page, "SOURCE_RECONSTRUCTION", block)
        else:
            page = page.replace('<!-- MAIN_EVIA_CHECK_START -->', '<!-- SOURCE_RECONSTRUCTION_START -->\n' + block + '\n<!-- SOURCE_RECONSTRUCTION_END -->\n<!-- MAIN_EVIA_CHECK_START -->', 1)
        page = page.replace('<a href="#main-evia-check">EVIA hiện tại</a>', '<a href="#source-reconstruction">Mã chính hiện tại</a>')
    target.write_text(page, encoding="utf-8")
    print(json.dumps({"html":str(target), "bytes":target.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__": main()
