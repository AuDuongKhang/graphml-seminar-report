"""Khởi tạo W&B dùng chung: mỗi thư mục output có một run id cố định để resume sau khi phiên bị ngắt."""
import os
import subprocess
import uuid
from pathlib import Path

DEFAULT_PROJECT = "graphml-seminar-report"


def stable_run_id(out_dir: Path) -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / "wandb_run_id.txt"
    if f.exists():
        return f.read_text().strip()
    rid = uuid.uuid4().hex[:12]
    f.write_text(rid)
    return rid


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def init_wandb(exp_id: str, config: dict, out_dir: Path, project: str | None = None,
               mode: str | None = None, group: str | None = None):
    import wandb

    cfg = dict(config, exp_id=exp_id, git_commit=git_commit())
    return wandb.init(
        project=project or os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT),
        entity=os.environ.get("WANDB_ENTITY") or None,
        name=exp_id,
        group=group,
        id=stable_run_id(Path(out_dir)),
        resume="allow",
        config=cfg,
        mode=mode or os.environ.get("WANDB_MODE") or None,
        dir=str(out_dir),
    )
