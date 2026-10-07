"""Phân loại lỗi như Fig 5 của bài. Định nghĩa của nhóm (bài không mô tả):

FE (format error)  : lời gọi LLM trả về văn bản nhưng không tách được danh sách `-xxx`
NR (no response)   : lời gọi rỗng / quá thời gian / lỗi API
IO (invalid output): khái niệm dự đoán không xuất hiện trong văn bản bài giảng (ảo giác, sai ngôn ngữ)
CE (concept error) : khái niệm dự đoán có trong văn bản nhưng không thuộc nhãn vàng
CO (concept omission): khái niệm vàng bị bỏ sót
"""
from collections import Counter

from .normalize import normalize_concept


def classify_errors(pred: dict, gold: dict, texts: dict, statuses: list) -> dict:
    c = Counter()
    for s in statuses:
        if s == "format_error":
            c["FE"] += 1
        elif s in ("no_response", "timeout", "api_error"):
            c["NR"] += 1
    for l, g in gold.items():
        text = normalize_concept(texts.get(l, ""))
        for x in pred.get(l, set()) - g:
            c["IO" if x not in text else "CE"] += 1
        c["CO"] += len(g - pred.get(l, set()))
    return {k: c[k] for k in ("FE", "NR", "IO", "CE", "CO")}
