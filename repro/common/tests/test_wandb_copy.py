"""Chép run W&B từ project team (không cho Public trên gói academic trial) sang project cá nhân công khai."""
import types

from repro.common import wandb_copy as wc


def _run(id_, name, state="finished", **kw):
    return types.SimpleNamespace(id=id_, name=name, state=state, **kw)


def test_select_runs_skips_unfinished_excluded_and_already_copied():
    runs = [_run("a1", "G1a-base-csqa-text"), _run("b2", "G1-csqa-curriculum-dryrun/stage2_qa"),
            _run("c3", "G4-csqa-graph-first-eval", state="crashed"), _run("d4", "G1b-graphvis-csqa-image"),
            _run("e5", "G1-csqa-curriculum/stage1_graph_reading")]
    todo, skipped = wc.select_runs(runs, exclude_regex="dryrun", exclude_ids={"e5"}, existing_ids={"d4"})
    assert [r.id for r in todo] == ["a1"]
    assert {i: why for i, _, why in skipped} == {"b2": "loại theo tên", "c3": "trạng thái crashed",
                                                  "d4": "đã có ở đích", "e5": "loại theo id"}


def test_clean_row_drops_internal_keys_and_media_refs():
    row = {"_step": 3, "_runtime": 9.1, "train/loss": 0.4, "predictions": {"_type": "table-file", "path": "x"},
           "examples": {"_type": "images/separated", "filenames": ["a.png"]}, "accuracy_by_nodes/0-10": 0.8}
    assert wc.clean_row(row) == {"train/loss": 0.4, "accuracy_by_nodes/0-10": 0.8}


def test_table_key_from_artifact_name():
    assert wc.table_key("run-3522cf6077b0-predictions:v0") == "predictions"
    assert wc.table_key("run-abc123-accuracy_by_nodes:v2") == "accuracy_by_nodes"


class _Table:
    def __init__(self, columns=None, data=None):
        self.columns, self.data = columns, data


class _NewRun:
    def __init__(self, **kw):
        self.init, self.logged, self.summary, self.finished = kw, [], {}, False

    def log(self, data, step=None):
        self.logged.append((step, data))

    def finish(self):
        self.finished = True


def test_copy_run_keeps_id_name_group_history_tables_and_summary():
    made = []
    fake_wandb = types.SimpleNamespace(Table=_Table, init=lambda **kw: made.append(_NewRun(**kw)) or made[-1],
                                       Image=lambda *a, **k: ("img", a, k))
    table = _Table(columns=["id", "pred_letter"], data=[["q1", "A"]])
    art = types.SimpleNamespace(type="run_table", name="run-x1-predictions:v0", get=lambda key: table)
    src = types.SimpleNamespace(
        id="x1", name="G1b-graphvis-csqa-image", group="G1", job_type=None, tags=[], notes=None,
        config={"mode": "image"}, summary=types.SimpleNamespace(_json_dict={
            "accuracy": 0.81, "_runtime": 5, "predictions": {"_type": "table-file"}}),
        scan_history=lambda: [{"_step": 0, "accuracy_by_nodes/0-10": 0.86, "predictions": {"_type": "table-file"}}],
        logged_artifacts=lambda: [art])
    wc.copy_run(src, "team", "me", "proj", wandb_mod=fake_wandb)
    new = made[0]
    assert new.init["id"] == "x1" and new.init["entity"] == "me" and new.init["project"] == "proj"
    assert new.init["name"] == "G1b-graphvis-csqa-image" and new.init["group"] == "G1"
    assert new.init["config"]["mode"] == "image" and new.init["config"]["copied_from"] == "team/proj/x1"
    assert (0, {"accuracy_by_nodes/0-10": 0.86}) in new.logged
    logged_tables = [d["predictions"] for _, d in new.logged if "predictions" in d]
    assert logged_tables[0].columns == ["id", "pred_letter"] and logged_tables[0].data == [["q1", "A"]]
    assert new.summary == {"accuracy": 0.81} and new.finished
