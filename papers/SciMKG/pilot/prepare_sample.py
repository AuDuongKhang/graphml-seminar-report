"""Chỉ tải Example.zip nhỏ của tác giả; không tải RawData.zip hoặc toàn bộ tập."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

URL = "https://zenodo.org/records/19691725/files/Example.zip"
EXPECTED_MD5 = "42ba1f64364b888fec957175a7a7395f"
MAX_BYTES = 2_000_000


def inspect_archive(archive: Path):
    raw = archive.read_bytes()
    if len(raw) > MAX_BYTES or hashlib.md5(raw).hexdigest() != EXPECTED_MD5:
        raise ValueError("Tệp mẫu không khớp bản công bố; kiểm tra lại nguồn trước khi dùng")
    with zipfile.ZipFile(archive) as z:
        if sum(i.file_size for i in z.infolist()) > 20_000_000:
            raise ValueError("Kích thước giải nén vượt giới hạn mẫu")
        for entry in z.infolist():
            path = Path(entry.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Đường dẫn ZIP không an toàn")
        if z.testzip() is not None:
            raise ValueError("Tệp ZIP hỏng")
        return {"source_url": URL, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                "files": [{"path": i.filename, "bytes": i.file_size} for i in z.infolist() if not i.is_dir()]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, default=Path(__file__).parent / "data" / "Example.zip")
    p.add_argument("--download", action="store_true", help="Tải đúng tệp mẫu nhỏ nếu chưa có")
    p.add_argument("--extract-to", type=Path, help="Chỉ giải nén tệp mẫu nhỏ sau khi kiểm tra hash và CRC")
    args = p.parse_args()
    if not args.archive.exists():
        if not args.download:
            p.error("Chưa có Example.zip; dùng --download hoặc cung cấp tệp mẫu đã tải")
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(URL, headers={"User-Agent": "SciMKG-student-pilot/1.0"})
        with urllib.request.urlopen(request, timeout=20) as response:
            data = response.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise ValueError("Phản hồi vượt giới hạn tệp mẫu")
        partial = args.archive.with_suffix(".partial.zip")
        partial.write_bytes(data)
        inspect_archive(partial)
        partial.replace(args.archive)
    inventory = inspect_archive(args.archive)
    if args.extract_to:
        args.extract_to.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.archive) as z:
            z.extractall(args.extract_to)
    print(json.dumps(inventory, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
