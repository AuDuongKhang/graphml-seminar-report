import json
from types import SimpleNamespace

import pytest

from gv.eval.complexity import load_predictions


def test_load_predictions_prefers_local_file(tmp_path):
    d = tmp_path / "G1a-base-csqa-text"
    d.mkdir()
    (d / "predictions.jsonl").write_text(json.dumps({"id": "q1", "n_nodes": 3, "gold": "A", "pred": "B"}))
    assert load_predictions("G1a-base-csqa-text", tmp_path, api=None) == [{"id": "q1", "n_nodes": 3, "gold": "A",
                                                                         "pred": "B"}]


class _Art:
    def __init__(self, name, type_, table):
        self.name, self.type, self._table = name, type_, table

    def get(self, key):
        assert key == "predictions"
        return self._table


def _run(name, state, created, rows):
    table = SimpleNamespace(columns=["id", "n_nodes", "gold", "pred_letter", "raw"], data=rows)
    arts = [_Art("run-x-examples:v0", "run_table", None), _Art("run-x-predictions:v0", "run_table", table)]
    return SimpleNamespace(name=name, state=state, created_at=created, logged_artifacts=lambda: arts)


class _Api:
    default_entity = "team"

    def __init__(self, runs):
        self._runs, self.calls = runs, []

    def runs(self, path, filters=None):
        self.calls.append((path, filters))
        return self._runs


def test_load_predictions_falls_back_to_latest_finished_wandb_run(tmp_path, monkeypatch):
    """Lượt Kaggle sau không còn predictions.jsonl của lượt trước: đọc bảng predictions đã log lên W&B."""
    monkeypatch.setenv("WANDB_PROJECT", "proj")
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    api = _Api([_run("e", "finished", "2026-10-15T20:00:00", [["old", 1, "A", "A", "A"]]),
                _run("e", "crashed", "2026-10-15T23:00:00", [["crash", 1, "A", "A", "A"]]),
                _run("e", "finished", "2026-10-15T21:00:00", [["q1", 12, "C", None, "?"]])])
    rows = load_predictions("e", tmp_path, api=api)
    assert rows == [{"id": "q1", "n_nodes": 12, "gold": "C", "pred": None}]
    assert api.calls == [("team/proj", {"display_name": "e"})]


def test_load_predictions_without_any_source_names_the_experiment(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_PROJECT", "proj")
    with pytest.raises(FileNotFoundError, match="G7-lowres50-eval"):
        load_predictions("G7-lowres50-eval", tmp_path, api=_Api([]))


# ---------------- Lượt 9: C2 hết VRAM ở anyres-672, chạy lại phải bỏ qua điểm đã xong ----------------
def _finished(name, acc, created="2026-10-20T10:00:00"):
    return SimpleNamespace(name=name, state="finished", created_at=created,
                           summary=SimpleNamespace(_json_dict={} if acc is None else {"accuracy": acc}))


def test_resolution_sweep_skips_points_already_on_wandb(monkeypatch):
    from gv.eval import complexity, mcqa
    from gv.model import load

    monkeypatch.setenv("WANDB_PROJECT", "proj")
    done = {"C2-graphvis-pad-50": 0.82, "C2-graphvis-pad-orig": 0.81, "C2-graphvis-anyres-50": 0.82}

    class Api(_Api):
        def runs(self, path, filters=None):
            name = filters["display_name"]
            out = [_finished(name, done[name])] if name in done else []
            if name == "C2-graphvis-anyres-orig":            # run hỏng nhưng W&B vẫn ghi finished: không có accuracy
                out = [_finished(name, None)]
            return out

    loads, evals = [], []
    monkeypatch.setattr(load, "load_model_and_processor", lambda **kw: loads.append(kw["image_mode"]) or ("m", "p"))
    monkeypatch.setattr(mcqa, "eval_mcqa", lambda *a, **kw: evals.append(a[4]) or {"accuracy": 0.5})
    res = complexity.resolution_sweep("adp", "C2-graphvis", sizes=(50,), image_modes=("pad", "anyres"), api=Api([]))
    assert evals == ["C2-graphvis-anyres-orig"]
    assert loads == ["anyres"]                                # chế độ pad đã đủ điểm: không nạp mô hình
    assert res == {"C2-graphvis-pad-50": 0.82, "C2-graphvis-pad-orig": 0.81, "C2-graphvis-anyres-50": 0.82,
                   "C2-graphvis-anyres-orig": 0.5}
