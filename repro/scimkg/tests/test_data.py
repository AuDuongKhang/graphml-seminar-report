from smkg.data import parse_author_blocks, parse_integration, parse_segments, stratified_lessons
from smkg.normalize import normalize_concept


def test_normalize_fullwidth_and_spaces():
    assert normalize_concept(" 细胞膜 ") == normalize_concept("细胞膜") == "细胞膜"
    assert normalize_concept("ＤＮＡ") == "DNA"
    assert normalize_concept("- 叶绿体。") == "叶绿体"
    assert normalize_concept("“上皮 组织”") == "上皮组织"


def test_parse_segments():
    txt = '"00:11-01:18": "各位同学大家好"\n\n"01:18-03:13": "1959年，科学家"\n'
    assert parse_segments(txt) == [("00:11-01:18", "各位同学大家好"), ("01:18-03:13", "1959年，科学家")]


VALIDATION = ('00:06-01:23\r\n肺 环境 蝾螈 \r\n11 2 16 \r\nhello, "环境" ："Could refer to ..."\r\n'
              '"发育" ："..."\r\n\r\n\r\n01:23-02:12\r\n雄蛙 抱队 \r\n1 1 \r\nhello,\r\n1 1 \r\n\r\n')


def test_parse_author_blocks_with_multiline_feedback():
    blocks = parse_author_blocks(VALIDATION)
    assert blocks == [("00:06-01:23", {"肺": 11, "环境": 2, "蝾螈": 16}), ("01:23-02:12", {"雄蛙": 1, "抱队": 1})]


def test_parse_integration():
    assert parse_integration("01:23-02:12 鸣叫\n02:12-03:38 槟榔\n\n") == [("01:23-02:12", "鸣叫"), ("02:12-03:38", "槟榔")]


def test_stratified_lessons_balances_subjects():
    manifest = {f"b{i}": "biology" for i in range(10)} | {f"p{i}": "physics" for i in range(10)} | \
               {f"c{i}": "chemistry" for i in range(10)}
    picked = stratified_lessons(manifest, 6, seed=0)
    assert len(picked) == 6 and sum(1 for l in picked if l.startswith("b")) == 2
    assert stratified_lessons(manifest, None) == sorted(manifest)
