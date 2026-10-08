import shutil

import pytest
from PIL import Image

from gv.data.render import pad_to_square_resize
from gv.data.retrieve import SubGraph


def test_pad_to_square_resize_makes_square():
    img = Image.new("RGB", (300, 100), "white")
    out = pad_to_square_resize(img, 50)
    assert out.size == (50, 50)


@pytest.mark.skipif(shutil.which("dot") is None, reason="cần Graphviz (dot)")
def test_render_graph_writes_png(tmp_path):
    from gv.data.render import render_graph

    sg = SubGraph(nodes=["a", "b"], triples=[("a", "is a", "b")], qc={"a"}, ac={"b"})
    p = render_graph(sg, tmp_path / "g.png")
    assert p.exists() and Image.open(p).size[0] > 10
    p50 = render_graph(sg, tmp_path / "g50.png", size_px=50)
    assert Image.open(p50).size == (50, 50)


@pytest.mark.skipif(shutil.which("dot") is None, reason="cần Graphviz (dot)")
def test_render_empty_graph_placeholder(tmp_path):
    from gv.data.render import render_graph

    p = render_graph(SubGraph(nodes=[], triples=[], qc=set(), ac=set()), tmp_path / "e.png")
    assert p.exists()
