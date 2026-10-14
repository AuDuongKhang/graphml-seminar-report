"""Chụp nguồn Wikipedia cho khái niệm dự đoán; giữ URL, hash và lỗi rõ ràng."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.parse import quote
import urllib.request

from bs4 import BeautifulSoup
from pilot import digest, write_json, normalize


def snapshot_wikipedia(concept, language, title, output):
    url = f"https://{language}.wikipedia.org/wiki/" + quote(title.replace(" ", "_"), safe="")
    target = output / (digest(url.encode())[:20] + ".html")
    try:
        if target.exists():
            raw = target.read_bytes()
        else:
            request = urllib.request.Request(url, headers={"User-Agent": "SciMKG-student-pilot/1.0 (local educational research)"})
            with urllib.request.urlopen(request, timeout=25) as response:
                raw = response.read(2_000_001)
                url = response.url
            if len(raw) > 2_000_000: raise ValueError("Trang nguồn vượt giới hạn")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        soup = BeautifulSoup(raw, "html.parser")
        bodies = soup.select(".mw-parser-output")
        if not bodies: raise ValueError("Không có nội dung bài Wikipedia")
        body = max(bodies, key=lambda item: len(item.get_text()))
        for element in body.select("table, .hatnote, .shortdescription, sup.reference, .navbox, .infobox"):
            element.decompose()
        paragraphs = [p.get_text(" ", strip=True) for p in body.find_all("p")]
        text = "\n".join(p for p in paragraphs if len(p) > 45)[:4500]
        if not text: raise ValueError("Không có đoạn văn nguồn")
        return {"concept": normalize(concept), "url": url, "text": text,
                "retrieved_at": datetime.now(timezone.utc).isoformat(), "sha256": digest(raw),
                "snapshot_path": str(target.resolve()), "mapped_title": title}
    except Exception as error:
        return {"concept": normalize(concept), "url": url, "error": str(error)}


def collect(mapping, language, destination):
    titles = list(dict.fromkeys(mapping.values()))
    with ThreadPoolExecutor(max_workers=4) as pool:
        fetched = list(pool.map(lambda title: snapshot_wikipedia(title, language, title, destination / "snapshots"), titles))
    by_title = dict(zip(titles, fetched))
    records = [{**by_title[title], "concept": normalize(concept)} for concept, title in mapping.items()]
    write_json(destination / "wikipedia.json", [x for x in records if "text" in x])
    write_json(destination / "source-status.json", records)
    return records


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mapping", type=Path, help="JSON: khái niệm dự đoán → tên bài Wikipedia được kiểm tra")
    parser.add_argument("--language", choices=["zh", "vi", "en"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = collect(json.loads(args.mapping.read_text()), args.language, args.out)
    print(json.dumps({"sources": sum("text" in x for x in records), "errors": [x for x in records if "error" in x]}, ensure_ascii=False))
