"""Vẽ đồ thị con thành ảnh PNG bằng Graphviz (bố cục `dot`, không có tham số học)."""
from pathlib import Path

from PIL import Image

PLACEHOLDER = "no relevant knowledge"


def pad_to_square_resize(img: Image.Image, size: int, fill=(255, 255, 255)) -> Image.Image:
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(img.convert("RGB"), ((side - w) // 2, (side - h) // 2))
    return canvas.resize((size, size), Image.LANCZOS)


def render_graph(sg, out_png: Path, size_px: int | None = None, dpi: int = 96) -> Path:
    """size_px: nếu có, đệm ảnh thành vuông rồi thu về size_px x size_px (thí nghiệm ảnh độ phân giải thấp)."""
    import graphviz

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    g = graphviz.Digraph(format="png", graph_attr={"dpi": str(dpi)}, node_attr={"shape": "ellipse"})
    if not sg.nodes:
        g.node(PLACEHOLDER)
    for n in sg.nodes:
        g.node(n)
    for h, r, t in sg.triples:
        g.edge(h, t, label=r)
    stem = out_png.with_suffix("")
    g.render(filename=str(stem), cleanup=True)   # tạo <stem>.png
    produced = stem.with_suffix(".png")
    if produced != out_png:
        produced.replace(out_png)
    if size_px:
        pad_to_square_resize(Image.open(out_png), size_px).save(out_png)
    return out_png
