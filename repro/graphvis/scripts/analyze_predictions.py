"""Phân tích bảng predictions tải từ W&B (Export CSV, lưu thành <thư mục>/<tên run viết thường>/predictions.csv):
McNemar giữa các mô hình CSQA, độ chính xác theo số nút, MMBench theo nhóm câu và vị trí đáp án, G6 theo tác vụ.
Chạy: python -X utf8 repro/graphvis/scripts/analyze_predictions.py wandb
"""
import collections
import csv
import math
import re
import sys
from pathlib import Path

D = Path(sys.argv[1])
WORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                     "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}


def load(name):
    return list(csv.DictReader(open(D / name / "predictions.csv", encoding="utf-8")))


def mcnemar(a, b):
    """a, b: dict id -> đúng/sai. Trả (chỉ a đúng, chỉ b đúng, p hai phía, kiểm định nhị thức chính xác)."""
    ids = a.keys() & b.keys()
    n10 = sum(a[i] and not b[i] for i in ids)
    n01 = sum(b[i] and not a[i] for i in ids)
    n, k = n10 + n01, min(n10, n01)
    p = min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0
    return n10, n01, p, len(ids)


def csqa(name):
    return {r["id"]: r["pred_letter"] == r["gold"] for r in load(name)}


print("=== CSQA dev: độ chính xác và McNemar")
M = {k: csqa(k) for k in ["g1a-base-csqa-text", "g1a-base-csqa-image", "g1b-graphvis-csqa-image",
                          "c3-text-only-eval", "g4-csqa-joint-eval"]}
for k, v in M.items():
    print(f"  {k:26s} {100 * sum(v.values()) / len(v):6.2f}  n={len(v)}")
for a, b in [("g1b-graphvis-csqa-image", "c3-text-only-eval"), ("g1b-graphvis-csqa-image", "g4-csqa-joint-eval"),
             ("g1a-base-csqa-text", "g1a-base-csqa-image"), ("g1b-graphvis-csqa-image", "g1a-base-csqa-text"),
             ("c3-text-only-eval", "g1a-base-csqa-text"), ("g4-csqa-joint-eval", "c3-text-only-eval")]:
    n10, n01, p, n = mcnemar(M[a], M[b])
    print(f"  {a} vs {b}: chỉ trái đúng {n10}, chỉ phải đúng {n01}, p={p:.4f}")

print("=== CSQA theo số nút (n câu; gốc-text / gốc-ảnh / GraphVis / C3 / joint)")
nodes = {r["id"]: int(r["n_nodes"]) for r in load("g1b-graphvis-csqa-image")}
bins = [(0, 10, "0-10"), (11, 20, "11-20"), (21, 40, "21-40"), (41, 10**9, ">40")]
for lo, hi, lab in bins:
    ids = [i for i, n in nodes.items() if lo <= n <= hi]
    accs = [100 * sum(M[k][i] for i in ids) / len(ids) for k in M]
    print(f"  {lab:6s} n={len(ids):4d} " + " ".join(f"{x:5.1f}" for x in accs))
    n10, n01, p, _ = mcnemar({i: M['g1b-graphvis-csqa-image'][i] for i in ids}, {i: M['c3-text-only-eval'][i] for i in ids})
    print(f"         GraphVis vs C3: {n10}/{n01}, p={p:.3f}")
empty = [i for i, n in nodes.items() if n == 0]
print("  đồ thị rỗng:", len(empty))

print("=== Ảnh đổi đúng/sai của mô hình gốc: text->image")
t, im = M["g1a-base-csqa-text"], M["g1a-base-csqa-image"]
print("  đúng->sai", sum(t[i] and not im[i] for i in t), " sai->đúng", sum(im[i] and not t[i] for i in t))
for name in ["g1a-base-csqa-text", "g1a-base-csqa-image", "g1b-graphvis-csqa-image", "c3-text-only-eval"]:
    rows = load(name)
    print(f"  {name}: pred {dict(sorted(collections.Counter(r['pred_letter'] for r in rows).items()))}")
print("  gold", dict(sorted(collections.Counter(r["gold"] for r in load("c3-text-only-eval")).items())))

print("=== MMBench: gốc vs GraphVis")
mb = {k: load(k) for k in ["g2-base-mmbench", "g2-graphvis-mmbench"]}
for k, rows in mb.items():
    ok = [r["correct"] == "true" for r in rows]
    raw = collections.Counter(r["raw"].strip() for r in rows)
    byg = {L: round(100 * sum(r["correct"] == "true" for r in rows if r["gold"] == L) /
                    sum(r["gold"] == L for r in rows), 1) for L in "ABCD"}
    print(f"  {k}: single {100 * sum(ok) / len(ok):.2f}; pred {dict(sorted(raw.items()))}; theo đáp án đúng {byg}")
cat = collections.defaultdict(lambda: [[], []])
for j, k in enumerate(mb):
    for r in mb[k]:
        cat[r["category"]][j].append(r["correct"] == "true")
print("  nhóm câu (n, gốc, GraphVis, chênh):")
for c, (a, b) in sorted(cat.items(), key=lambda x: sum(x[1][1]) / len(x[1][1]) - sum(x[1][0]) / len(x[1][0])):
    pa, pb = 100 * sum(a) / len(a), 100 * sum(b) / len(b)
    print(f"    {c:40s} {len(a):4d} {pa:5.1f} {pb:5.1f} {pb - pa:+6.1f}")
ka = {r["index"]: r["correct"] == "true" for r in mb["g2-base-mmbench"]}
kb = {r["index"]: r["correct"] == "true" for r in mb["g2-graphvis-mmbench"]}
print("  McNemar single-pass gốc vs GraphVis:", mcnemar(ka, kb))

print("=== ScienceQA-IMG")
sq = {k: load(k) for k in ["g2-base-scienceqa", "g2-graphvis-scienceqa"]}
for k, rows in sq.items():
    raw = collections.Counter(r["raw"].strip()[:12] for r in rows)
    bad = sum(not re.fullmatch(r"[A-E]", r["raw"].strip()) for r in rows)
    print(f"  {k}: n={len(rows)} acc={100 * sum(r['correct'] == 'true' for r in rows) / len(rows):.2f} "
          f"không phải 1 chữ cái: {bad}; top {raw.most_common(6)}")
a = {i: r["correct"] == "true" for i, r in enumerate(sq["g2-base-scienceqa"])}
b = {i: r["correct"] == "true" for i, r in enumerate(sq["g2-graphvis-scienceqa"])}
same_gold = all(x["gold"] == y["gold"] for x, y in zip(*sq.values()))
print("  cùng thứ tự câu:", same_gold, " McNemar:", mcnemar(a, b))

print("=== G6 đọc ảnh")
for k in ["g6-base", "g6-after-stage1"]:
    rows = load(k)
    print(f"  {k}:")
    for task in ["node_count", "edge_count", "node_degree", "highest_degree", "node_description", "triple_listing"]:
        rs = [r for r in rows if r["task"] == task]
        sc = sum(float(r["score"]) for r in rs) / len(rs)
        dig = sum(bool(re.search(r"\d", r["pred"])) for r in rs)
        words = sum(any(w in re.findall(r"[a-z]+", r["pred"].lower()) for w in WORDS) for r in rs)
        fmt = sum(bool(re.search(r"\(.+?,.+?,.+?\)", r["pred"])) for r in rs)
        line = f"    {task:17s} n={len(rs):3d} điểm={100 * sc:5.1f} có chữ số={dig:3d} có số bằng chữ={words:3d}"
        if task in ("node_count", "edge_count"):
            errs = []
            for r in rs:
                g = int(re.search(r"\d+", r["gold"]).group())
                m = re.search(r"-?\d+", r["pred"])
                if m:
                    errs.append(int(m.group()) - g)
            if errs:
                ae = sorted(abs(e) for e in errs)
                line += (f" | lệch tuyệt đối trung vị={ae[len(ae) // 2]} ; lệch ±1: {sum(x <= 1 for x in ae)}/{len(ae)}"
                         f" ; đếm thiếu {sum(e < 0 for e in errs)} / thừa {sum(e > 0 for e in errs)}")
        if task == "triple_listing":
            line += f" | có dạng (h, r, t): {fmt}"
        print(line)
    ex = [r["pred"][:110] for r in rows if r["task"] == "node_count"][:3]
    print("    ví dụ node_count:", ex)
