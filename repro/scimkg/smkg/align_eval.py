"""Đánh giá căn chỉnh khái niệm-ảnh (Table 7): CLIPScore, BLIP-ITM, (tùy chọn) VQAScore, và phiếu chấm người.

Cặp (khái niệm, ảnh) lấy từ SciMKG.zip (trường `png` của từng khái niệm). Tên ảnh được đánh số toàn cục theo môn
(`N.png`), trùng với tên file trong RawData. CLIP/BLIP chỉ hiểu tiếng Anh nên khái niệm được dịch bằng LLM (có cache).
X-VLM-ITM không có bản HF nên không chạy (ghi trong DEVIATIONS).
"""
import csv
import json
import random
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .llm import parse_concepts
from .prompts import IMAGE_ALIGN, SUBJECT_NAME, fill

KG_FILES = {"biology": "biology/bioloy.json", "physics": "physics/physics.json", "chemistry": "chemical/Chemical.json"}
TRANSLATE = ("Translate the following Chinese K-12 {subject} terms into English, keeping the same order. "
             "Output one term per line starting with '- ' and nothing else.\n{terms}")


@dataclass(frozen=True)
class Pair:
    concept: str
    image: Path
    subject: str


def build_image_index(manifest: dict, lessons_dir: Path) -> dict:
    """(môn, 'N.png') -> đường dẫn; tên trùng giữa các bài thì lấy bài đứng trước theo thứ tự tên."""
    idx = {}
    for lesson in sorted(manifest):
        for p in sorted((Path(lessons_dir) / lesson).glob("*.png")):
            idx.setdefault((manifest[lesson], p.name), p)
    return idx


def load_pairs(kg_zip: Path, image_index: dict) -> list:
    pairs = []
    with zipfile.ZipFile(kg_zip) as zf:
        for subject, member in KG_FILES.items():
            if member not in zf.namelist():
                continue
            for c in json.loads(zf.read(member)):
                for png in c.get("png") or []:
                    path = image_index.get((subject, png))
                    if png != "none" and path is not None:
                        pairs.append(Pair(concept=c["concept"], image=path, subject=subject))
    return pairs


def parse_translations(text: str, n: int) -> list:
    items, _ = parse_concepts(text)
    return (items + [""] * n)[:n]


def translate(concepts: list, subject: str, client, batch: int = 40) -> dict:
    out = {}
    for i in range(0, len(concepts), batch):
        chunk = concepts[i:i + batch]
        reply = client.chat(fill(TRANSLATE, subject=SUBJECT_NAME.get(subject, subject), terms="\n".join(chunk)))
        out.update(zip(chunk, parse_translations(reply.text, len(chunk))))
    return out


def clipscore_from_cos(cos: float) -> float:
    """CLIPScore của Hessel et al. (2021): 2,5 · max(cos, 0)."""
    return round(2.5 * max(cos, 0.0), 10)


def clip_scores(pairs: list, texts: list, model_id: str = "openai/clip-vit-large-patch14", batch: int = 32) -> list:
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, proc = CLIPModel.from_pretrained(model_id).to(dev).eval(), CLIPProcessor.from_pretrained(model_id)
    cos = []
    for i in range(0, len(pairs), batch):
        imgs = [Image.open(p.image).convert("RGB") for p in pairs[i:i + batch]]
        enc = proc(text=texts[i:i + batch], images=imgs, return_tensors="pt", padding=True, truncation=True).to(dev)
        with torch.inference_mode():
            out = model(**enc)
        ie = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        te = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        cos += (ie * te).sum(-1).tolist()
    return cos


def blip_itm(pairs: list, texts: list, model_id: str = "Salesforce/blip-itm-large-coco", batch: int = 16) -> list:
    import torch
    from PIL import Image
    from transformers import BlipForImageTextRetrieval, BlipProcessor

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = BlipForImageTextRetrieval.from_pretrained(model_id).to(dev).eval()
    proc = BlipProcessor.from_pretrained(model_id)
    probs = []
    for i in range(0, len(pairs), batch):
        imgs = [Image.open(p.image).convert("RGB") for p in pairs[i:i + batch]]
        enc = proc(images=imgs, text=texts[i:i + batch], return_tensors="pt", padding=True).to(dev)
        with torch.inference_mode():
            logits = model(**enc, use_itm_head=True).itm_score
        probs += torch.softmax(logits, dim=-1)[:, 1].tolist()
    return probs


def vqascore(pairs: list, texts: list, model: str = "clip-flant5-xl") -> list:
    import t2v_metrics  # pip install t2v-metrics (tùy chọn, cần GPU lớn)
    scorer = t2v_metrics.VQAScore(model=model)
    return [float(scorer(images=[str(p.image)], texts=[t]).item()) for p, t in zip(pairs, texts)]


def regenerate_alignment(image: Path, lesson_concepts: list, subject: str, client, runs: int = 3) -> list:
    """Căn chỉnh lại một ảnh bằng prompt C1-C4 của tác giả, gọi `runs` lần và lấy hợp (như mã gốc)."""
    prompt = fill(IMAGE_ALIGN, subject=SUBJECT_NAME.get(subject, subject), concepts=lesson_concepts)
    found = []
    for i in range(runs):
        items, _ = parse_concepts(client.chat(prompt, image=image, seed=42 + i).text)
        found += [x for x in items if x.lower() != "none" and x not in found]
    return found


def make_human_sheet(pairs: list, n: int, out_dir: Path, seed: int = 42, translations: dict | None = None) -> Path:
    """CSV cho 2 người chấm (0 = không khớp ... 1 = khớp hoàn toàn) + thư mục ảnh đi kèm."""
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    picked = random.Random(seed).sample(pairs, min(n, len(pairs)))
    f = out_dir / "human_sheet.csv"
    with open(f, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pair_id", "concept_zh", "concept_en", "image_file", "rater_1", "rater_2"])
        for i, p in enumerate(picked):
            name = f"{i:03d}_{p.image.name}"
            shutil.copy(p.image, out_dir / "images" / name)
            w.writerow([i, p.concept, (translations or {}).get(p.concept, ""), name, "", ""])
    return f


def score_human_sheet(path: Path) -> dict:
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8")) if r["rater_1"] != "" and r["rater_2"] != ""]
    a = [float(r["rater_1"]) for r in rows]
    b = [float(r["rater_2"]) for r in rows]
    n = len(rows)
    mean = (sum(a) + sum(b)) / (2 * n) if n else 0.0
    ba, bb = [x >= 0.5 for x in a], [x >= 0.5 for x in b]
    po = sum(x == y for x, y in zip(ba, bb)) / n if n else 0.0
    pa, pb = sum(ba) / n if n else 0.0, sum(bb) / n if n else 0.0
    pe = pa * pb + (1 - pa) * (1 - pb)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return {"mean": mean, "mean_rater_1": sum(a) / n if n else 0.0, "mean_rater_2": sum(b) / n if n else 0.0,
            "cohen_kappa": kappa, "n": n}
