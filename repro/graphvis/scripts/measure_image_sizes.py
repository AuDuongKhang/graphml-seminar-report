"""Đo kích thước ảnh Graphviz trong một split đã đóng gói (data/<scale>/<dataset>/<split>.tar trên HUB_REPO),
đọc thẳng header PNG, không giải nén. Chạy: python -X utf8 repro/graphvis/scripts/measure_image_sizes.py <split.tar>
"""
import json
import struct
import sys
import tarfile

FONT_PX = 14 * 96 / 72          # Graphviz mặc định 14 pt, render ở 96 dpi -> cỡ chữ ~18,7 px
sizes, nodes = {}, {}
with tarfile.open(sys.argv[1]) as t:
    for m in t:
        if not m.isfile():
            continue
        if m.name.startswith("images/") and m.name.endswith(".png"):
            head = t.extractfile(m).read(24)
            assert head[:8] == b"\x89PNG\r\n\x1a\n"
            w, h = struct.unpack(">II", head[16:24])
            sizes[m.name[len("images/"):-4]] = (w, h)
        elif m.name == "graphs.jsonl":
            for line in t.extractfile(m):
                d = json.loads(line)
                nodes[d["id"]] = len(d["nodes"])


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


ids = [i for i in sizes if i in nodes and nodes[i] > 0]
print("ảnh:", len(sizes), "có đồ thị không rỗng:", len(ids))
W = [sizes[i][0] for i in ids]
H = [sizes[i][1] for i in ids]
L = [max(sizes[i]) for i in ids]
for name, xs in [("rộng", W), ("cao", H), ("cạnh dài", L)]:
    print(f"  {name:9s} trung vị {q(xs, .5)}  p10 {q(xs, .1)}  p90 {q(xs, .9)}  max {max(xs)}")
font = [FONT_PX * 336 / l for l in L]
print(f"cỡ chữ sau khi thu về 336 px: trung vị {q(font, .5):.1f}  p10 {q(font, .1):.1f}  p90 {q(font, .9):.1f}  min {min(font):.1f}")
print("tỉ lệ ảnh có chữ < 4 px:", round(100 * sum(f < 4 for f in font) / len(font), 1), "%;  < 6 px:",
      round(100 * sum(f < 6 for f in font) / len(font), 1), "%")
print("ảnh rộng hơn cao:", round(100 * sum(w > h for w, h in zip(W, H)) / len(W), 1), "%; tỉ lệ rộng/cao trung vị",
      round(q([w / h for w, h in zip(W, H)], .5), 2))
for lo, hi, lab in [(1, 10, "1-10"), (11, 20, "11-20"), (21, 40, "21-40"), (41, 10**9, ">40")]:
    sel = [i for i in ids if lo <= nodes[i] <= hi]
    ls = [max(sizes[i]) for i in sel]
    fs = [FONT_PX * 336 / l for l in ls]
    print(f"  {lab:6s} n={len(sel):4d} cạnh dài trung vị {q(ls, .5):5d}  cỡ chữ trung vị {q(fs, .5):4.1f} px")
