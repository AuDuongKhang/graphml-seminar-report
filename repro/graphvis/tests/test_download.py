import json

import pytest

from gv.data.download import QAItem, parse_qa_line, group_grounded

CSQA_LINE = json.dumps({"answerKey": "A", "id": "q1", "question": {
    "question_concept": "revolving door", "stem": "A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?",
    "choices": [{"label": "A", "text": "bank"}, {"label": "B", "text": "library"}, {"label": "C", "text": "department store"},
                {"label": "D", "text": "mall"}, {"label": "E", "text": "new york"}]}})


def test_parse_qa_line_csqa():
    it = parse_qa_line(CSQA_LINE)
    assert it == QAItem(id="q1", question=json.loads(CSQA_LINE)["question"]["stem"],
                        choices=["bank", "library", "department store", "mall", "new york"],
                        labels=["A", "B", "C", "D", "E"], answer="A")


def test_group_grounded_by_question():
    items = [parse_qa_line(CSQA_LINE)]
    rows = [{"sent": "... bank.", "ans": c, "qc": ["revolving_door"], "ac": [c.replace(" ", "_")]} for c in items[0].choices]
    groups = group_grounded(items, rows)
    assert len(groups) == 1 and len(groups[0]) == 5


def test_group_grounded_rejects_misaligned_rows():
    items = [parse_qa_line(CSQA_LINE)]
    rows = [{"sent": "x", "ans": "bank", "qc": [], "ac": []}] * 4
    with pytest.raises(ValueError):
        group_grounded(items, rows)
