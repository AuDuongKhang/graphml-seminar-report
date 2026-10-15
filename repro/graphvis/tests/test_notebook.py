"""Notebook GraphVis_Repro.ipynb (sinh từ make_notebook.py) phải clone được repo private trên Kaggle/Colab."""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "GraphVis_Repro.ipynb"


def _cells():
    return ["".join(c["source"]) for c in json.loads(NB.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


def test_secrets_are_readable_before_cloning():
    cells = _cells()
    define = next(i for i, c in enumerate(cells) if "def secret(" in c)
    clone = next(i for i, c in enumerate(cells) if '"clone"' in c)
    assert define <= clone
    if define == clone:
        assert cells[clone].index("def secret(") < cells[clone].index('"clone"')


def test_clone_uses_github_token_without_writing_it_to_disk():
    clone_cell = next(c for c in _cells() if '"clone"' in c)
    assert 'secret("GITHUB_TOKEN")' in clone_cell
    assert "http.extraheader" in clone_cell            # token chỉ đi qua header, không nằm trong URL/.git/config
    assert "@github.com" not in clone_cell


def test_secret_helper_defined_once():
    assert sum("def secret(" in c for c in _cells()) == 1
