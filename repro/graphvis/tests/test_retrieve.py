from gv.data.retrieve import CPNet, build_question_graph, load_cpnet_csv

NAMES = ["revolving_door", "security", "bank", "door", "library", "building", "money"]
TRIPLES = [("revolving_door", "is_a", "door"), ("door", "at_location", "bank"),
           ("security", "at_location", "bank"), ("door", "part_of", "building"),
           ("library", "is_a", "building"), ("money", "related_to", "bank"),
           ("money", "at_location", "bank")]


def toy():
    return CPNet.from_triples(NAMES, TRIPLES)


def test_load_cpnet_csv_applies_qagnn_pruning(tmp_path):
    vocab = tmp_path / "concept.txt"
    vocab.write_text("a\nb\nc\npeople\n", encoding="utf-8")
    csv = tmp_path / "conceptnet.en.csv"
    csv.write_text(
        "isa\ta\tb\t1.0\n"            # giữ
        "isa\ta\tb\t2.0\n"            # trùng -> bỏ
        "hascontext\ta\tc\t1.0\n"     # hascontext -> bỏ
        "atlocation\ta\tpeople\t1.0\n"  # blacklist -> bỏ
        "relatedto\tc\tc\t1.0\n"      # vòng tự thân -> bỏ
        "usedfor\tc\tb\t1.0\n",       # giữ
        encoding="utf-8")
    cp = load_cpnet_csv(csv, vocab, cache=tmp_path / "cp.npz")
    assert cp.relations(0, 1) == [5]      # isa
    assert cp.relations(2, 1) == [16]     # usedfor
    assert cp.relations(0, 2) == [] and cp.relations(1, 0) == []
    assert cp.neighbors(1) == {0, 2}
    cp2 = load_cpnet_csv(csv, vocab, cache=tmp_path / "cp.npz")  # nạp lại từ cache
    assert cp2.neighbors(1) == {0, 2} and cp2.relations(0, 1) == [5]


def test_two_hop_path_adds_intermediate():
    sg = build_question_graph(toy(), [{"qc": ["revolving_door", "security"], "ac": ["bank"]}])
    assert "door" in sg.nodes
    assert ("revolving door", "is a", "door") in sg.triples
    assert ("door", "at location", "bank") in sg.triples
    assert ("security", "at location", "bank") in sg.triples


def test_merges_all_choices():
    sg = build_question_graph(toy(), [{"qc": ["revolving_door"], "ac": ["bank"]},
                                      {"qc": ["revolving_door"], "ac": ["building"]}])
    assert {"bank", "building"} <= set(sg.nodes)


def test_related_to_dropped_when_specific_relation_exists():
    sg = build_question_graph(toy(), [{"qc": ["money"], "ac": ["bank"]}])
    assert sg.triples == [("money", "at location", "bank")]


def test_concepts_without_paths_are_not_drawn():
    sg = build_question_graph(toy(), [{"qc": ["revolving_door", "library"], "ac": ["bank"]}])
    assert "library" not in sg.nodes


def test_empty_grounding_yields_placeholder_graph():
    sg = build_question_graph(toy(), [{"qc": [], "ac": []}])
    assert sg.nodes == [] and sg.triples == []


def test_unknown_concepts_are_ignored():
    sg = build_question_graph(toy(), [{"qc": ["not_in_kg"], "ac": ["bank"]}])
    assert sg.nodes == []


def test_hub_concepts_are_skipped():
    # "hub" nối với 3 nút khác -> bậc 4; với max_hub_degree=3 không được làm nút trung gian
    names = ["q", "a", "hub", "x1", "x2", "spec"]
    cp = CPNet.from_triples(names, [("q", "related_to", "hub"), ("hub", "related_to", "a"),
                                    ("hub", "related_to", "x1"), ("hub", "related_to", "x2"),
                                    ("q", "is_a", "spec"), ("spec", "at_location", "a")])
    sg = build_question_graph(cp, [{"qc": ["q"], "ac": ["a"]}], max_hub_degree=3, paths_per_pair=10)
    assert "hub" not in sg.nodes and "spec" in sg.nodes


def test_paths_per_pair_keeps_most_specific_intermediate():
    names = ["q", "a", "generic", "specific", "g1", "g2"]
    cp = CPNet.from_triples(names, [("q", "related_to", "generic"), ("generic", "related_to", "a"),
                                    ("generic", "related_to", "g1"), ("generic", "related_to", "g2"),
                                    ("q", "is_a", "specific"), ("specific", "at_location", "a")])
    sg = build_question_graph(cp, [{"qc": ["q"], "ac": ["a"]}], paths_per_pair=1)
    assert set(sg.nodes) == {"q", "a", "specific"}


def test_max_nodes_keeps_question_and_answer_concepts():
    sg = build_question_graph(toy(), [{"qc": ["revolving_door", "security"], "ac": ["bank"]}], max_nodes=3)
    assert set(sg.nodes) == {"revolving door", "security", "bank"}
    assert all(h in sg.nodes and t in sg.nodes for h, _, t in sg.triples)
