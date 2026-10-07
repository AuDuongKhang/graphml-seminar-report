from repro.common.tracking import stable_run_id


def test_run_id_is_stable_across_restarts(tmp_path):
    a = stable_run_id(tmp_path)
    b = stable_run_id(tmp_path)
    assert a == b and len(a) >= 8
    assert (tmp_path / "wandb_run_id.txt").read_text().strip() == a


def test_run_id_differs_between_output_dirs(tmp_path):
    assert stable_run_id(tmp_path / "a") != stable_run_id(tmp_path / "b")
