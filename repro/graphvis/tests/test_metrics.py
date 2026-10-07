from gv.data.retrieve import SubGraph
from gv.eval.graph_reading import score_graph_task
from gv.eval.mcqa import accuracy_by_bins, extract_choice_letter, insert_kaping, node_bin

SG = SubGraph(nodes=["choker", "jewelry", "box", "store"],
              triples=[("choker", "is a", "jewelry"), ("jewelry", "at location", "box"), ("box", "used for", "store")],
              qc={"choker"}, ac={"store"})


def test_extract_choice_letter_variants():
    assert extract_choice_letter("B", 5) == "B"
    assert extract_choice_letter("The answer is (C).", 5) == "C"
    assert extract_choice_letter("b. bank", 5) == "B"
    assert extract_choice_letter("A", 4) == "A"
    assert extract_choice_letter("E", 4) is None
    assert extract_choice_letter("", 5) is None
    assert extract_choice_letter("I think it is a bank", 5) is None


def test_node_bins_and_accuracy_by_bins():
    assert [node_bin(n) for n in (0, 10, 11, 20, 21, 40, 41)] == ["0-10", "0-10", "11-20", "11-20", "21-40", "21-40", ">40"]
    acc = accuracy_by_bins([5, 15, 15], [True, False, True])
    assert acc == {"0-10": 1.0, "11-20": 0.5}


def test_insert_kaping_puts_facts_before_question():
    p = "The image represents ... useful. Question: Where?\nA. x\nB. y"
    out = insert_kaping(p, "FACTS\n(a, b, c)\n")
    assert out.index("FACTS") < out.index("Question: Where?") and out.endswith("B. y")
    q = "Question: Where?\nA. x"
    assert insert_kaping(q, "F\n").startswith("F\nQuestion: Where?")
    assert insert_kaping(q, "") == q


def test_graph_count_scores():
    assert score_graph_task("node_count", "There are 4 nodes.", SG, "There are 4 nodes in the graph.") == 1.0
    assert score_graph_task("node_count", "There are 5 nodes.", SG, "There are 4 nodes in the graph.") == 0.0
    assert score_graph_task("edge_count", "3", SG, "There are 3 edges in the graph.") == 1.0
    assert score_graph_task("node_count", "many", SG, "There are 4 nodes in the graph.") == 0.0


def test_graph_degree_scores():
    gold = 'The degree of the node "jewelry" is 2.'
    assert score_graph_task("node_degree", "It has 2 connections.", SG, gold) == 1.0
    assert score_graph_task("highest_degree", 'One node is "box" with a degree of 2.', SG, "") == 1.0
    assert score_graph_task("highest_degree", 'One node is "store" with a degree of 2.', SG, "") == 0.0
    assert score_graph_task("highest_degree", 'One node is "box" with a degree of 3.', SG, "") == 0.0


def test_graph_listing_scores_are_partial():
    s = score_graph_task("triple_listing", "(choker, is a, jewelry)", SG, "")
    assert abs(s - 1 / 3) < 1e-9
    assert score_graph_task("node_description", "nodes: Choker, box", SG, "") == 0.5
