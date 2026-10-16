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


# ---------------- Chạy nền theo lượt (Save Version) ----------------
EXPERIMENTS = {"smoke", "G0", "G1a", "G1b", "G1c", "G2", "G3", "G4joint", "G4graph", "G4ocr", "G5", "G6", "G7",
               "C1", "C2", "C3"}


def _guarded(cell):
    import re
    m = re.match(r'if want\("([A-Za-z0-9]+)"\):', cell.strip())
    return m.group(1) if m else None


def test_every_experiment_cell_is_guarded_by_exactly_one_id():
    exp_cells = [c for c in _cells() if ("gv(" in c or "sh(" in c) and "def gv(" not in c]
    ids = [_guarded(c) for c in exp_cells]
    assert None not in ids, "mọi ô gọi gv.cli phải mở đầu bằng if want(...)"
    assert set(ids) == EXPERIMENTS and len(ids) == len(EXPERIMENTS)


def test_batches_cover_each_experiment_once_in_dependency_order():
    config = next(c for c in _cells() if "BATCHES = " in c)
    ns = {}
    exec(config[config.index("BATCHES = "):config.index("# end BATCHES")], ns)
    batches = ns["BATCHES"]
    flat = [e for b in sorted(batches) for e in batches[b]]
    assert sorted(flat) == sorted(EXPERIMENTS)
    order = {e: i for i, e in enumerate(flat)}
    for before, after in [("G0", "G1a"), ("G1b", "G6"), ("G1b", "G2"), ("G1b", "G5"), ("G4joint", "G3"), ("G4joint", "G5"),
                          ("G1b", "C1"), ("C3", "C1"), ("G7", "C1"), ("G1b", "C2"), ("smoke", "G0")]:
        assert order[before] < order[after], (before, after)


def test_commands_stop_the_batch_on_failure():
    helpers = next(c for c in _cells() if "def gv(" in c)
    assert "_exit_code" in helpers and "raise" in helpers
    assert not any(line.lstrip().startswith("!python -m gv.cli") for c in _cells() for line in c.splitlines())


# ---------------- /kaggle/working trống ở mỗi lần Save Version ----------------
def test_batches_without_g0_rebuild_data_without_logging_to_wandb():
    g0 = next(c for c in _cells() if c.strip().startswith('if want("G0")'))
    head, _, rest = g0.partition("\nelse:")
    assert rest, "lượt không có G0 phải dựng lại dữ liệu (lệnh train/eval không tự dựng)"
    assert "build-data" in rest and "--no_wandb" in rest
    assert "--no_wandb" not in head                     # lượt 1 vẫn ghi run G0 lên W&B


def test_later_batches_require_hub_repo_for_adapters():
    setup = next(c for c in _cells() if '"clone"' in c)
    assert "HUB_REPO" in setup and "BATCH" in setup and "raise" in setup


def test_existing_clone_is_switched_to_the_configured_branch():
    setup = next(c for c in _cells() if '"clone"' in c)
    assert '"pull"' not in setup                        # pull giữ nhánh cũ khi đổi BRANCH
    assert '"fetch"' in setup and '"checkout"' in setup and "FETCH_HEAD" in setup
