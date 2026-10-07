"""Đọc file .zip từ xa qua HTTP Range: chỉ tải các member cần dùng thay vì cả archive (QA-GNN 4,2 GB; SciMKG RawData 5,4 GB)."""
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable


class HttpRangeFile(io.RawIOBase):
    def __init__(self, url: str):
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            self.url = r.geturl()
            self.size = int(r.headers["Content-Length"])
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, off, whence=0):
        if whence == 0:
            self.pos = off
        elif whence == 1:
            self.pos += off
        else:
            self.pos = self.size + off
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def open_remote_zip(url: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BufferedReader(HttpRangeFile(url), buffer_size=1 << 20))


def decoded_name(info: zipfile.ZipInfo) -> str:
    """Tên member; archive tiếng Trung không đặt cờ UTF-8 thì giải mã lại theo GBK."""
    if info.flag_bits & 0x800:
        return info.filename
    try:
        return info.filename.encode("cp437").decode("gbk")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return info.filename


def extract_members(url_or_zip, predicate: Callable[[str], bool], dest: Path,
                    strip_prefix: str = "") -> list[Path]:
    """Trích các member có predicate(tên đã giải mã) đúng vào dest; bỏ qua file đã tồn tại cùng kích thước."""
    zf = url_or_zip if isinstance(url_or_zip, zipfile.ZipFile) else open_remote_zip(url_or_zip)
    dest = Path(dest)
    out = []
    for info in zf.infolist():
        name = decoded_name(info)
        if info.is_dir() or not predicate(name):
            continue
        rel = name[len(strip_prefix):] if strip_prefix and name.startswith(strip_prefix) else name
        target = dest / rel
        if target.exists() and target.stat().st_size == info.file_size:
            out.append(target)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 20)
        out.append(target)
    return out
