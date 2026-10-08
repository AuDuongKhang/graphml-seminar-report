"""Chuẩn hóa tên khái niệm trước khi so khớp với nhãn vàng (NFKC, bỏ khoảng trắng và dấu câu bao quanh)."""
import re
import unicodedata

_EDGE = re.compile(r"^[\s\-•*·\"'“”‘’「」『』《》()（）\[\]【】.,，。;；:：!！?？]+|[\s\-•*·\"'“”‘’「」『』《》()（）\[\]【】.,，。;；:：!！?？]+$")


def normalize_concept(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = _EDGE.sub("", s.strip())
    return re.sub(r"\s+", "", s)
