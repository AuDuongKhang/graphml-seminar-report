"""Dữ liệu SciMKG (Zenodo record 19691725): nhãn vàng cho ~100 bài giảng + dữ liệu thô của đúng các bài đó.

Chỉ tải chọn lọc qua HTTP Range từ RawData.zip (5,4 GB): file đoạn phụ đề có mốc thời gian, ảnh slide,
và kết quả trung gian EVIA của tác giả (_Extractionner, _Validationner, _Intergrationner).
"""
import json
import random
import re
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from repro.common.remote_zip import decoded_name, open_remote_zip

from .normalize import normalize_concept

RECORD = "https://zenodo.org/api/records/19691725/files"
RAW_URL = f"{RECORD}/RawData.zip/content"
GT_URL = f"{RECORD}/Groundtruth.zip/content"
KG_URL = f"{RECORD}/SciMKG.zip/content"
SUBJECTS = {"biology": "biology", "physics": "physics", "chemcial": "chemistry", "chemical": "chemistry",
            "chemistry": "chemistry"}
AUTHOR_SUFFIXES = ("_Extractionner", "_Validationner", "_Intergrationner")
_TIME = re.compile(r"^\d{2}:\d{2}-\d{2}:\d{2}$")
_SEG = re.compile(r'"(\d{2}:\d{2}-\d{2}:\d{2})"\s*:\s*"(.*?)"\s*$')


@dataclass
class Lesson:
    name: str
    subject: str
    segments: list                      # [(mốc thời gian, văn bản)]
    images: list                        # [Path]
    gt: set                             # khái niệm vàng (đã chuẩn hóa)
    author_extraction: dict = field(default_factory=dict)   # time -> {concept: 1}
    author_validation: dict = field(default_factory=dict)   # time -> {concept: điểm phiếu (tối đa 16)}
    author_integration: list = field(default_factory=list)  # [(time, concept)]


def parse_segments(text: str) -> list:
    out = []
    for line in text.splitlines():
        m = _SEG.match(line.strip())
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def parse_author_blocks(text: str) -> list:
    """Định dạng _Extractionner/_Validationner: khối bắt đầu bằng dòng mốc thời gian, dòng khái niệm, dòng điểm."""
    lines = [l.rstrip("\r") for l in text.split("\n")]
    blocks, i = [], 0
    while i < len(lines):
        if _TIME.match(lines[i].strip()) and i + 2 < len(lines):
            concepts = lines[i + 1].split()
            scores = lines[i + 2].split()
            if scores and all(s.lstrip("-").isdigit() for s in scores) and len(scores) == len(concepts):
                blocks.append((lines[i].strip(), {c: int(s) for c, s in zip(concepts, scores)}))
                i += 3
                continue
        i += 1
    return blocks


def parse_integration(text: str) -> list:
    out = []
    for line in text.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2 and _TIME.match(parts[0]):
            out.append((parts[0], parts[1].strip()))
    return out


def stratified_lessons(manifest: dict, n: int | None, seed: int = 42) -> list:
    """manifest: bài -> môn. Lấy n bài chia đều theo môn (seed cố định); n=None -> mọi bài."""
    names = sorted(manifest)
    if not n or n >= len(names):
        return names
    by_subj = {}
    for l in names:
        by_subj.setdefault(manifest[l], []).append(l)
    rng = random.Random(seed)
    subjects = sorted(by_subj)
    picked = []
    for k, s in enumerate(subjects):
        quota = n // len(subjects) + (1 if k < n % len(subjects) else 0)
        picked += rng.sample(by_subj[s], min(quota, len(by_subj[s])))
    return sorted(picked)


def _download(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
            f.write(r.read())
    return dest


def load_groundtruth(gt_zip: Path) -> dict:
    """bài -> tập khái niệm vàng (chuẩn hóa). Một số bài có 2 file -> hợp lại."""
    gt = {}
    with zipfile.ZipFile(gt_zip) as zf:
        for info in zf.infolist():
            name = decoded_name(info)
            if name.endswith("groundtruth.txt"):
                lesson = name.split("/")[1]
                text = zf.read(info).decode("utf-8", "replace")
                gt.setdefault(lesson, set()).update(normalize_concept(l) for l in text.splitlines() if l.strip())
    return {k: {c for c in v if c} for k, v in gt.items()}


def download_subset(dest: Path, with_images: bool = True) -> Path:
    """Tải nhãn vàng + dữ liệu thô của các bài có nhãn. Ghi manifest.json (bài -> môn).
    with_images=False: bỏ ảnh slide (~1 GB) khi chỉ chạy trích xuất khái niệm."""
    dest = Path(dest)
    gt_zip = _download(GT_URL, dest / "Groundtruth.zip")
    gt = load_groundtruth(gt_zip)
    manifest_path = dest / "manifest.json"
    if manifest_path.exists() and (not with_images or any((dest / "lessons").rglob("*.png"))):
        return dest
    zf = open_remote_zip(RAW_URL)
    manifest = {}
    for info in zf.infolist():
        parts = decoded_name(info).split("/")
        if len(parts) >= 3 and parts[-2] in gt:
            manifest.setdefault(parts[-2], SUBJECTS.get(parts[1].lower(), parts[1].lower()))

    def wanted(name: str) -> bool:
        parts = name.split("/")
        if len(parts) < 3 or parts[-2] not in manifest:
            return False
        f, lesson = parts[-1], parts[-2]
        is_png = with_images and f.lower().endswith(".png")
        return f == f"{lesson}.txt" or is_png or f in {lesson + s for s in AUTHOR_SUFFIXES}

    # đặt mọi file của một bài vào lessons/<bài>/
    for info in zf.infolist():
        name = decoded_name(info)
        if info.is_dir() or not wanted(name):
            continue
        parts = name.split("/")
        target = dest / "lessons" / parts[-2] / parts[-1]
        if target.exists() and target.stat().st_size == info.file_size:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(zf.read(info))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return dest


def load_lessons(dest: Path, names: list | None = None) -> list:
    dest = Path(dest)
    manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    gt = load_groundtruth(dest / "Groundtruth.zip")
    out = []
    for name in names or sorted(manifest):
        d = dest / "lessons" / name
        read = lambda suffix: (d / f"{name}{suffix}").read_text(encoding="utf-8", errors="replace") \
            if (d / f"{name}{suffix}").exists() else ""
        out.append(Lesson(
            name=name, subject=manifest[name], segments=parse_segments(read(".txt")),
            images=sorted(d.glob("*.png"), key=lambda p: (len(p.stem), p.stem)), gt=gt.get(name, set()),
            author_extraction=dict(parse_author_blocks(read("_Extractionner"))),
            author_validation=dict(parse_author_blocks(read("_Validationner"))),
            author_integration=parse_integration(read("_Intergrationner"))))
    return out
