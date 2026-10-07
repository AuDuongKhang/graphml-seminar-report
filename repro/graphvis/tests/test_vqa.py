from gv.eval.complexity import combine_bins
from gv.eval.vqa import (mmbench_circular_accuracy, mmbench_prompt, pope_metrics, pope_yes, sample_mmbench_groups,
                         scienceqa_prompt)


def test_mmbench_circular_accuracy():
    rows = [{"index": 5, "correct": True}, {"index": 1_000_005, "correct": True},
            {"index": 7, "correct": True}, {"index": 1_000_007, "correct": False}]
    assert mmbench_circular_accuracy(rows) == 0.5


def test_sample_mmbench_groups_keeps_circular_variants_together():
    rows = [{"index": i} for i in (1, 2, 3, 1_000_001, 1_000_002, 2_000_001)]
    picked = sample_mmbench_groups(rows, n_groups=1, seed=0)
    groups = {r["index"] % 1_000_000 for r in picked}
    assert len(groups) == 1
    g = groups.pop()
    assert len(picked) == sum(1 for r in rows if r["index"] % 1_000_000 == g)


def test_mmbench_prompt_skips_nan_options_and_hint():
    row = {"question": "Q?", "hint": "nan", "A": "x", "B": "y", "C": "nan", "D": "nan"}
    assert mmbench_prompt(row) == "Q?\nA. x\nB. y\nAnswer with the option's letter from the given choices directly."
    row["hint"] = "H"
    assert mmbench_prompt(row).startswith("H\nQ?")


def test_scienceqa_prompt():
    row = {"question": "Which?", "choices": ["a", "b"], "hint": "ctx"}
    assert scienceqa_prompt(row) == ("Context: ctx\nWhich?\nA. a\nB. b\n"
                                     "Answer with the option's letter from the given choices directly.")
    row["hint"] = ""
    assert scienceqa_prompt(row).startswith("Which?")


def test_pope_yes_parsing_and_metrics():
    assert pope_yes("Yes, there is.") and not pope_yes("No.") and not pope_yes("There is not a dog.")
    m = pope_metrics(["yes", "no", "yes", "no"], ["yes", "yes", "no", "no"])
    assert m["acc"] == 0.5 and m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5
    assert m["yes_ratio"] == 0.5


def test_combine_bins():
    rows = {"base": [{"n_nodes": 5, "gold": "A", "pred": "A"}, {"n_nodes": 50, "gold": "A", "pred": "B"}],
            "graphvis": [{"n_nodes": 5, "gold": "A", "pred": "A"}, {"n_nodes": 50, "gold": "A", "pred": "A"}]}
    table = combine_bins(rows)
    assert ["base", "0-10", 1.0, 1] in table and ["base", ">40", 0.0, 1] in table
    assert ["graphvis", ">40", 1.0, 1] in table
