"""Dữ liệu G0 lưu trên HF Hub: lượt Kaggle sau tải về thay vì dựng lại (/kaggle/working trống mỗi lần Save Version)."""
import shutil
from pathlib import Path

from huggingface_hub.utils import EntryNotFoundError

from gv.data import hub


def _split(root: Path) -> Path:
    d = root / "csqa" / "train"
    (d / "images").mkdir(parents=True)
    (d / "images" / "q1.png").write_bytes(b"png")
    (d / "dv.jsonl").write_text('{"id": "q1", "image": "images/q1.png"}\n')
    (d / "stats.json").write_text("{}")
    (d / "wandb_run" / "wandb").mkdir(parents=True)
    (d / "wandb_run" / "wandb" / "debug.log").write_text("log")
    return d


class _Api:
    def __init__(self):
        self.files, self.repos = {}, []

    def create_repo(self, repo_id, private, exist_ok):
        self.repos.append((repo_id, private))

    def repo_info(self, repo_id):
        from huggingface_hub.utils import RepositoryNotFoundError
        if repo_id not in [r for r, _ in self.repos]:
            raise RepositoryNotFoundError(repo_id)

    def file_exists(self, repo_id, filename):
        return filename in self.files

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id):
        self.files[path_in_repo] = Path(path_or_fileobj).read_bytes()

    def hf_hub_download(self, repo_id, filename):
        if filename not in self.files:
            raise EntryNotFoundError(filename)
        p = Path(self.tmp) / "dl.tar"
        p.write_bytes(self.files[filename])
        return str(p)


def test_archive_path_separates_scale_and_tag():
    assert hub.archive_path("lite", "csqa", "train") == "data/lite/csqa/train.tar"
    assert hub.archive_path("lite", "csqa", "dev_lowres50") == "data/lite/csqa/dev_lowres50.tar"


def test_push_then_pull_restores_split_without_wandb_files(tmp_path):
    api = _Api()
    api.tmp = tmp_path
    src = _split(tmp_path / "a")
    hub.push("me/repo", "data/lite/csqa/train.tar", src, api=api)
    assert api.repos == [("me/repo", True)]                        # repo private
    assert not list(src.parent.glob("*.tar"))                      # file tạm đã xóa

    dst = tmp_path / "b" / "csqa" / "train"
    assert hub.pull("me/repo", "data/lite/csqa/train.tar", dst, api=api)
    assert (dst / "images" / "q1.png").read_bytes() == b"png"
    assert (dst / "dv.jsonl").read_text() == (src / "dv.jsonl").read_text()
    assert not (dst / "wandb_run").exists()                        # run id cũ không được mang sang


def test_push_skips_when_archive_already_on_hub(tmp_path):
    api = _Api()
    api.files["data/lite/csqa/train.tar"] = b"old"
    hub.push("me/repo", "data/lite/csqa/train.tar", _split(tmp_path), api=api)
    assert api.files["data/lite/csqa/train.tar"] == b"old"


def test_pull_returns_false_when_hub_has_no_copy(tmp_path):
    api = _Api()
    api.tmp = tmp_path
    assert not hub.pull("me/repo", "data/lite/csqa/train.tar", tmp_path / "x", api=api)


def test_build_data_uses_hub_copy_without_rebuilding(tmp_path, monkeypatch):
    from gv import pipeline

    monkeypatch.setattr(pipeline, "data_root", lambda: tmp_path)
    calls = []

    def fake_pull(repo, remote, dst, api=None):
        calls.append((repo, remote))
        shutil.copytree(_split(tmp_path / "src"), dst)
        return True

    monkeypatch.setattr(pipeline.hub, "pull", fake_pull)
    monkeypatch.setattr(pipeline.hub, "ensure_repo", lambda repo, api=None: None)
    monkeypatch.setattr(pipeline.dl, "download_qagnn", lambda *a: (_ for _ in ()).throw(AssertionError("rebuilt")))
    out = pipeline.build_data("csqa", "train", "lite", use_wandb=False, hub_repo="me/repo")
    assert calls == [("me/repo", "data/lite/csqa/train.tar")]
    assert (out / "stats.json").exists()


class _Forbidden(_Api):
    def create_repo(self, repo_id, private, exist_ok):
        from huggingface_hub.utils import HfHubHTTPError
        raise HfHubHTTPError("403 Forbidden: You don't have the rights to create a model under the namespace")


def test_ensure_repo_explains_namespace_and_token(tmp_path):
    import pytest
    with pytest.raises(RuntimeError, match="AuDuongKhang/graphvis-adapters.*HF_TOKEN"):
        hub.ensure_repo("AuDuongKhang/graphvis-adapters", api=_Forbidden())


def test_build_data_checks_hub_access_before_rebuilding(tmp_path, monkeypatch):
    """Log lượt 2: dựng lại 5 phút rồi mới lỗi 403 lúc đẩy lên Hub. Phải dừng trước khi tải/dựng."""
    import pytest

    from gv import pipeline

    monkeypatch.setattr(pipeline, "data_root", lambda: tmp_path)
    monkeypatch.setattr(pipeline.hub, "ensure_repo", lambda repo, api=None: (_ for _ in ()).throw(RuntimeError("403")))
    monkeypatch.setattr(pipeline.dl, "download_qagnn", lambda *a: (_ for _ in ()).throw(AssertionError("rebuilt")))
    with pytest.raises(RuntimeError, match="403"):
        pipeline.build_data("csqa", "train", "lite", use_wandb=False, hub_repo="wrong/repo")


def test_training_checks_hub_access_before_loading_anything(monkeypatch):
    """Upload adapter chạy sau nhiều giờ train: sai HUB_REPO/quyền phải lỗi ngay từ đầu."""
    import pytest

    from gv import train

    monkeypatch.setattr(train.hub, "ensure_repo", lambda repo, api=None: (_ for _ in ()).throw(RuntimeError("403")))
    with pytest.raises(RuntimeError, match="403"):
        train.run_stages({"exp_id": "x", "hub_repo": "wrong/repo", "stages": []})


def test_ensure_repo_does_not_recreate_existing_repo():
    """Token fine-grained chỉ cấp quyền cho một repo có sẵn có thể không được gọi API tạo repo."""
    api = _Forbidden()
    api.repos.append(("duongkhang2100/graphvis-adapters", True))
    hub.ensure_repo("duongkhang2100/graphvis-adapters", api=api)
