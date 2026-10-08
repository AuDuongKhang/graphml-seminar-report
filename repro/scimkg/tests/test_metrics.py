import json
import math

from smkg.data import Lesson
from smkg.errors import classify_errors
from smkg.metrics import lesson_sets, set_prf, universe_metrics
from smkg.runner import score_run


def test_set_prf_micro():
    m = set_prf({"l1": {"a", "b"}, "l2": {"c"}}, {"l1": {"a"}, "l2": {"c", "d"}})
    assert m["precision"] == 2 / 3 and m["recall"] == 2 / 3 and abs(m["f1"] - 2 / 3) < 1e-12


def test_set_prf_empty_prediction():
    assert set_prf({"l": set()}, {"l": {"a"}}) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_universe_metrics_perfect():
    m = universe_metrics({"l": {"a"}}, {"l": {"a"}}, {"l": {"a", "x"}}, scores={"l": {"a": 0.9, "x": 0.1}})
    assert m["acc"] == 1.0 and m["mcc"] == 1.0 and m["auc"] == 1.0


def test_universe_metrics_binary_auc_is_balanced_accuracy():
    m = universe_metrics({"l": {"a", "x"}}, {"l": {"a", "b"}}, {"l": {"a", "b", "x", "y"}})
    # TP=a, FP=x, FN=b, TN=y
    assert m["acc"] == 0.5 and m["mcc"] == 0.0 and m["auc"] == 0.5


def test_universe_metrics_mcc_undefined_is_zero():
    m = universe_metrics({"l": set()}, {"l": set()}, {"l": {"x"}})
    assert m["mcc"] == 0.0 and not math.isnan(m["acc"])


def test_lesson_sets_union_over_segments():
    assert lesson_sets({"L": {"t1": {"a"}, "t2": {"b"}}}) == {"L": {"a", "b"}}


def test_classify_errors():
    c = classify_errors(pred={"L": {"细胞", "火星人", "叶片"}}, gold={"L": {"细胞", "叶绿体"}},
                        texts={"L": "细胞 和 叶片"}, statuses=["ok", "format_error", "no_response", "timeout"])
    assert c == {"FE": 1, "NR": 2, "IO": 1, "CE": 1, "CO": 1}


def test_score_run_on_saved_logs(tmp_path):
    run = tmp_path / "run"
    (run / "lessons").mkdir(parents=True)
    (run / "lessons" / "L.json").write_text(json.dumps({
        "candidates": {"t": ["a", "b"]}, "per_model": {"t": {"m": ["a", "b"]}}, "status": {"t": {"m": "ok"}},
        "votes": {"t": {"m": [["a"], ["a"]]}}}), encoding="utf-8")
    lessons = [Lesson(name="L", subject="biology", segments=[("t", "a b")], images=[], gt={"a"})]
    m = score_run(run, lessons, ["m"], K=2, alpha=0.7, variant="full")
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["n_lessons"] == 1


def test_roc_auc_with_ties():
    from smkg.metrics import roc_auc
    assert roc_auc([True, False], [0.9, 0.1]) == 1.0
    assert roc_auc([True, False], [0.5, 0.5]) == 0.5
    assert roc_auc([True, True, False, False], [0.8, 0.4, 0.6, 0.2]) == 0.75
