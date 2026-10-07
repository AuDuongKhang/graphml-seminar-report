import zipfile

from repro.common.remote_zip import extract_members


def test_extract_members_filters_and_strips_prefix(tmp_path):
    z = tmp_path / "a.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("root/keep/x.txt", "x")
        zf.writestr("root/skip/y.bin", "y")
    out = extract_members(zipfile.ZipFile(z), lambda n: n.endswith(".txt"), tmp_path / "o", strip_prefix="root/")
    assert [p.relative_to(tmp_path / "o").as_posix() for p in out] == ["keep/x.txt"]
    assert (tmp_path / "o" / "keep" / "x.txt").read_text() == "x"


def test_extract_members_skips_existing_same_size(tmp_path):
    z = tmp_path / "a.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("f.txt", "abc")
    dest = tmp_path / "o"
    (dest).mkdir()
    (dest / "f.txt").write_text("zzz")  # cùng kích thước -> không ghi đè
    extract_members(zipfile.ZipFile(z), lambda n: True, dest)
    assert (dest / "f.txt").read_text() == "zzz"
