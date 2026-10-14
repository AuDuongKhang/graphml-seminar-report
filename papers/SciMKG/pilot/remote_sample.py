"""Đọc chỉ mục ZIP bằng HTTP Range, chỉ lấy thành phần của một bài đã chọn."""
from __future__ import annotations

import io
import json
from pathlib import Path
import re
import urllib.request
import zipfile

from pilot import digest, write_json


class RangeReader(io.RawIOBase):
    def __init__(self, url, max_bytes=24_000_000, opener=urllib.request.urlopen, cache=None):
        self.url, self.max_bytes, self.opener = url, max_bytes, opener
        self.cache = cache
        self.position, self.transferred, self.requests = 0, 0, []
        data, total = self.fetch("bytes=-65558", 65558)
        self.size, self.tail = total, (total - len(data), data)

    def fetch(self, interval, limit):
        cached = self.cache / (digest((self.url + interval).encode()) + ".json") if self.cache else None
        if cached and cached.is_file() and hasattr(self, "size"):
            metadata = json.loads(cached.read_text())
            binary = cached.with_suffix(".bin")
            data = binary.read_bytes()
            if metadata["total"] == self.size and len(data) <= limit and digest(data) == metadata["sha256"]:
                self.requests.append({"range": interval, "bytes": 0, "cached_bytes": len(data), "sha256": digest(data)})
                return data, self.size
        request = urllib.request.Request(self.url, headers={"Range": interval, "User-Agent": "SciMKG-student-pilot/1.0"})
        with self.opener(request, timeout=30) as response:
            if response.status != 206:
                raise ValueError("Máy chủ không hỗ trợ Range; dừng để tránh tải toàn bộ ZIP")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if not match:
                raise ValueError("Phản hồi Range thiếu giới hạn hợp lệ")
            start, end, total = map(int, match.groups())
            expected = end - start + 1
            if expected > limit or self.transferred + expected > self.max_bytes:
                raise ValueError("Vượt giới hạn tải mẫu")
            if hasattr(self, "size") and total != self.size:
                raise ValueError("ZIP thay đổi trong lúc tải")
            if interval.startswith("bytes=-"):
                if end != total - 1 or expected != min(limit, total):
                    raise ValueError("Range phần cuối không khớp")
            elif interval != f"bytes={start}-{end}":
                raise ValueError("Máy chủ trả sai khoảng byte")
            data = response.read(expected + 1)
            if len(data) != expected:
                raise ValueError("Phản hồi Range bị cắt hoặc dài hơn yêu cầu")
        self.transferred += len(data)
        self.requests.append({"range": interval, "bytes": len(data), "sha256": digest(data)})
        if cached:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.with_suffix(".bin").write_bytes(data)
            write_json(cached, {"total": total, "sha256": digest(data)})
        return data, total

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if position < 0 or whence not in (0, 1, 2):
            raise ValueError("Vị trí ZIP không hợp lệ")
        self.position = position
        return position

    def read(self, size=-1):
        stop = self.size if size < 0 else min(self.size, self.position + size)
        if stop <= self.position: return b""
        if self.position >= self.tail[0]:
            data = self.tail[1][self.position - self.tail[0]:stop - self.tail[0]]
        else:
            data, _ = self.fetch(f"bytes={self.position}-{stop - 1}", stop - self.position)
        self.position = stop
        return data


def inventory(url, output):
    reader = RangeReader(url, cache=Path(__file__).parent / "data/range-cache")
    with zipfile.ZipFile(reader, metadata_encoding="gbk") as archive:
        entries = [{"path": x.filename, "bytes": x.file_size, "compressed_bytes": x.compress_size,
                    "crc32": f"{x.CRC:08x}"} for x in archive.infolist() if not x.is_dir()]
    write_json(output, {"url": url, "archive_bytes": reader.size, "transferred_bytes": reader.transferred,
                        "requests": reader.requests, "files": entries})
    return entries


def download_selected(url, names, destination):
    reader, selected = RangeReader(url, cache=Path(__file__).parent / "data/range-cache"), []
    with zipfile.ZipFile(reader, metadata_encoding="gbk") as archive:
        for name in names:
            member = archive.getinfo(name)
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or member.file_size > 2_000_000:
                raise ValueError("Thành phần không an toàn hoặc vượt phạm vi mẫu")
            data = archive.read(member)  # zipfile kiểm tra CRC của từng thành phần.
            target = destination / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            selected.append({"path": name, "bytes": len(data), "sha256": digest(data), "crc32": f"{member.CRC:08x}"})
    receipt = {"url": url, "archive_bytes": reader.size, "transferred_bytes": reader.transferred,
               "requests": reader.requests, "selected": selected,
               "integrity": "CRC và SHA-256 từng thành phần; không xác minh MD5 toàn bộ ZIP vì không tải toàn bộ"}
    write_json(destination / "download-receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", choices=["RawData.zip", "Groundtruth.zip"])
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--member", action="append", default=[])
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    url = "https://zenodo.org/records/19691725/files/" + args.archive
    entries = inventory(url, args.inventory)
    print(json.dumps({"entries": len(entries), "inventory": str(args.inventory)}, ensure_ascii=False))
    if args.member:
        if not args.destination: parser.error("--member cần --destination")
        print(json.dumps(download_selected(url, args.member, args.destination), ensure_ascii=False, indent=2))
