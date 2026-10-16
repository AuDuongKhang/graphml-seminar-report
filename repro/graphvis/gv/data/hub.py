"""Lưu dữ liệu G0 (một split = một file .tar) lên HF Hub, cùng repo với adapter.

Mỗi lần Save Version trên Kaggle bắt đầu với /kaggle/working trống: lượt sau tải bản đã dựng về (~1 phút)
thay vì truy hồi và vẽ lại (~10 phút GPU quota mỗi lượt). Đường dẫn ảnh trong jsonl là tương đối nên giải nén
ở máy khác vẫn dùng được.
"""
import tarfile
from pathlib import Path

SKIP = {"wandb_run"}          # run id W&B của lượt dựng gốc, không mang sang lượt sau
DONE = "stats.json"           # build_data ghi file này cuối cùng


def archive_path(scale: str, dataset: str, split_name: str) -> str:
    return f"data/{scale}/{dataset}/{split_name}.tar"


def _api(api):
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    return api


def ensure_repo(hub_repo: str, api=None) -> None:
    """Tạo (hoặc xác nhận) repo private trước khi làm việc tốn thời gian: sai namespace hay token thiếu quyền
    Write thì lỗi ngay, không phải sau khi dựng dữ liệu/huấn luyện xong."""
    from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

    api = _api(api)
    try:
        api.repo_info(hub_repo)                     # đã có (vd tạo tay, token chỉ cấp quyền repo đó): không tạo lại
        return
    except RepositoryNotFoundError:
        pass
    try:
        api.create_repo(hub_repo, private=True, exist_ok=True)
    except HfHubHTTPError as e:
        raise RuntimeError(f"Không ghi được vào HF Hub repo {hub_repo!r}. HUB_REPO phải là <tên người dùng HF>/<tên repo> "
                           f"(tên người dùng xem ở huggingface.co/settings/profile) và HF_TOKEN cần quyền Write "
                           f"vào repo của namespace đó. Lỗi gốc: {e}") from e


def pack(src: Path, tar_path: Path) -> None:
    src = Path(src)
    with tarfile.open(tar_path, "w") as t:          # PNG đã nén sẵn, không nén lại
        for p in sorted(src.rglob("*")):
            rel = p.relative_to(src)
            if p.is_file() and rel.parts[0] not in SKIP:
                t.add(p, arcname=rel.as_posix())


def unpack(tar_path: Path, dst: Path) -> None:
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as t:
        kw = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        t.extractall(dst, **kw)


def pull(hub_repo: str, path_in_repo: str, dst: Path, api=None) -> bool:
    """True nếu Hub có bản đầy đủ và đã giải nén vào dst."""
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    try:
        local = _api(api).hf_hub_download(repo_id=hub_repo, filename=path_in_repo)
    except (EntryNotFoundError, RepositoryNotFoundError):
        return False
    unpack(Path(local), dst)
    return (Path(dst) / DONE).exists()


def push(hub_repo: str, path_in_repo: str, src: Path, api=None) -> None:
    api = _api(api)
    ensure_repo(hub_repo, api=api)
    if api.file_exists(hub_repo, path_in_repo):
        return
    src = Path(src)
    tar = src.parent / f"{src.name}.upload.tar"
    try:
        pack(src, tar)
        api.upload_file(path_or_fileobj=str(tar), path_in_repo=path_in_repo, repo_id=hub_repo)
    finally:
        tar.unlink(missing_ok=True)
