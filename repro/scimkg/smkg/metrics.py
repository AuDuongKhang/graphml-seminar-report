"""Chỉ số trích xuất khái niệm. Bài không định nghĩa cách tính nên nhóm cố định như sau:

- P/R/F1: micro trên tập khái niệm của mọi bài (so khớp sau chuẩn hóa).
- ACC/MCC/AUC: trên vũ trụ U_l = vàng ∪ pool ∪ dự đoán của từng bài, pool = mọi ứng viên mà bước trích xuất
  sinh ra (mẫu âm = ứng viên không thuộc nhãn vàng). AUC dùng điểm tỉ lệ phiếu nếu có; dự đoán nhị phân thì
  AUC = balanced accuracy.
"""
import math


def roc_auc(y: list, s: list) -> float:
    """AUC = xác suất một mẫu dương có điểm cao hơn một mẫu âm (thống kê Mann-Whitney, xử lý hòa bằng hạng trung bình)."""
    order = sorted(range(len(s)), key=lambda i: s[i])
    ranks = [0.0] * len(s)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and s[order[j + 1]] == s[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    pos = sum(y)
    neg = len(y) - pos
    return (sum(r for r, t in zip(ranks, y) if t) - pos * (pos + 1) / 2) / (pos * neg)


def lesson_sets(per_segment: dict) -> dict:
    """{bài: {đoạn: set}} -> {bài: set}."""
    return {l: set().union(*segs.values()) if segs else set() for l, segs in per_segment.items()}


def set_prf(pred: dict, gold: dict) -> dict:
    tp = sum(len(pred.get(l, set()) & g) for l, g in gold.items())
    np_ = sum(len(pred.get(l, set())) for l in gold)
    ng = sum(len(g) for g in gold.values())
    p = tp / np_ if np_ else 0.0
    r = tp / ng if ng else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def universe_metrics(pred: dict, gold: dict, pool: dict, scores: dict | None = None) -> dict:
    y, yhat, s = [], [], []
    for l, g in gold.items():
        pr = pred.get(l, set())
        for c in sorted(g | pool.get(l, set()) | pr):
            y.append(c in g)
            yhat.append(c in pr)
            s.append((scores or {}).get(l, {}).get(c, 0.0))
    tp = sum(a and b for a, b in zip(y, yhat))
    tn = sum(not a and not b for a, b in zip(y, yhat))
    fp = sum(not a and b for a, b in zip(y, yhat))
    fn = sum(a and not b for a, b in zip(y, yhat))
    n = len(y)
    acc = (tp + tn) / n if n else 0.0
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    if scores is not None and any(y) and not all(y):
        auc = roc_auc(y, s)
    else:
        tpr = tp / (tp + fn) if tp + fn else 0.0
        tnr = tn / (tn + fp) if tn + fp else 0.0
        auc = (tpr + tnr) / 2
    return {"acc": acc, "mcc": mcc, "auc": auc, "n_universe": n}
