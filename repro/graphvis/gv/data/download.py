"""Tải dữ liệu GraphVis: chỉ trích các file cần thiết từ gói tiền xử lý của QA-GNN (~60 MB nén / 4,2 GB)."""
import json
from dataclasses import dataclass
from pathlib import Path

from repro.common.remote_zip import extract_members

QAGNN_URL = "https://nlp.stanford.edu/projects/myasu/QAGNN/data_preprocessed_release.zip"
PREFIX = "data_preprocessed_release/"

# Khóa -> đường dẫn tương đối trong archive (sau khi bỏ PREFIX)
FILES = {
    "concept_txt": "cpnet/concept.txt",
    "cpnet_csv": "cpnet/conceptnet.en.csv",
    "csqa_train": "csqa/train_rand_split.jsonl",
    "csqa_dev": "csqa/dev_rand_split.jsonl",
    "csqa_grounded_train": "csqa/grounded/train.grounded.jsonl",
    "csqa_grounded_dev": "csqa/grounded/dev.grounded.jsonl",
    "obqa_train": "obqa/OpenBookQA-V1-Sep2018/Data/Main/train.jsonl",
    "obqa_test": "obqa/OpenBookQA-V1-Sep2018/Data/Main/test.jsonl",
    "obqa_grounded_train": "obqa/grounded/train.grounded.jsonl",
    "obqa_grounded_test": "obqa/grounded/test.grounded.jsonl",
}

# split dùng trong thực nghiệm -> (file câu hỏi, file grounded)
SPLITS = {
    ("csqa", "train"): ("csqa_train", "csqa_grounded_train"),
    ("csqa", "dev"): ("csqa_dev", "csqa_grounded_dev"),
    ("obqa", "train"): ("obqa_train", "obqa_grounded_train"),
    ("obqa", "test"): ("obqa_test", "obqa_grounded_test"),
}


@dataclass(frozen=True)
class QAItem:
    id: str
    question: str
    choices: list
    labels: list
    answer: str


def download_qagnn(dest: Path) -> dict:
    dest = Path(dest)
    wanted = {PREFIX + rel for rel in FILES.values()}
    missing = [rel for rel in FILES.values() if not (dest / rel).exists()]
    if missing:
        extract_members(QAGNN_URL, lambda n: n in wanted, dest, strip_prefix=PREFIX)
    return {k: dest / rel for k, rel in FILES.items()}


def parse_qa_line(line: str) -> QAItem:
    """Định dạng gốc của CSQA và OBQA: {"id","answerKey","question":{"stem","choices":[{"label","text"}]}}."""
    d = json.loads(line)
    q = d["question"]
    return QAItem(id=d["id"], question=q["stem"],
                  choices=[c["text"] for c in q["choices"]],
                  labels=[c["label"] for c in q["choices"]],
                  answer=d.get("answerKey", ""))


def load_qa(paths: dict, dataset: str, split: str) -> list:
    qfile, _ = SPLITS[(dataset, split)]
    with open(paths[qfile], encoding="utf-8") as f:
        return [parse_qa_line(l) for l in f if l.strip()]


def load_grounded(paths: dict, dataset: str, split: str) -> list:
    _, gfile = SPLITS[(dataset, split)]
    with open(paths[gfile], encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def group_grounded(items: list, rows: list) -> list:
    """Chia các dòng grounded (một dòng cho mỗi cặp câu hỏi-lựa chọn, theo thứ tự) thành nhóm theo câu hỏi."""
    need = sum(len(it.choices) for it in items)
    if need != len(rows):
        raise ValueError(f"grounded rows ({len(rows)}) != sum of choices ({need}); file lệch thứ tự")
    groups, i = [], 0
    for it in items:
        g = rows[i:i + len(it.choices)]
        for choice, row in zip(it.choices, g):
            if row.get("ans", choice).strip().lower() != choice.strip().lower():
                raise ValueError(f"grounded row lệch lựa chọn ở câu {it.id}: {row.get('ans')!r} vs {choice!r}")
        groups.append(g)
        i += len(it.choices)
    return groups
