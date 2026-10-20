"""Chép run W&B sang project khác, giữ nguyên id, tên, group, config, lịch sử, bảng và summary.

Dùng khi project gốc nằm trong team không được để Public (gói academic trial chỉ cho "Team"): chép sang
project dưới tài khoản cá nhân rồi để Public. Giữ id nên link cũ chỉ đổi phần entity.

    python -m repro.common.wandb_copy --src_entity <team> --dst_entity <tài khoản> --project graphml-seminar-report
"""
import argparse
import re
import tempfile
from pathlib import Path

MEDIA_TYPES = {"table-file", "images/separated", "image-file", "plotly-file", "html-file", "joined-table",
               "partitioned-table", "histogram"}


def is_media(v) -> bool:
    return isinstance(v, dict) and v.get("_type") in MEDIA_TYPES


def clean_row(row: dict) -> dict:
    """Bỏ khóa nội bộ của W&B (_step, _runtime, ...) và tham chiếu tới bảng/ảnh (chép riêng)."""
    return {k: v for k, v in row.items() if not k.startswith("_") and not is_media(v)}


def table_key(artifact_name: str) -> str:
    """run-<id>-predictions:v0 -> predictions"""
    return re.sub(r"^run-[A-Za-z0-9]+-", "", artifact_name.split(":")[0])


def select_runs(runs, exclude_regex: str | None, exclude_ids: set, existing_ids: set):
    todo, skipped = [], []
    rx = re.compile(exclude_regex) if exclude_regex else None
    for r in runs:
        if r.state != "finished":
            skipped.append((r.id, r.name, f"trạng thái {r.state}"))
        elif r.id in exclude_ids:
            skipped.append((r.id, r.name, "loại theo id"))
        elif rx and rx.search(r.name or ""):
            skipped.append((r.id, r.name, "loại theo tên"))
        elif r.id in existing_ids:
            skipped.append((r.id, r.name, "đã có ở đích"))
        else:
            todo.append(r)
    return todo, skipped


def _images(src, value, wandb_mod, tmp: Path):
    """images/separated hoặc image-file -> wandb.Image (tải file media của run gốc)."""
    files = value.get("filenames") or [value.get("path")]
    captions = value.get("captions") or [value.get("caption")] * len(files)
    out = []
    for fn, cap in zip(files, captions):
        local = src.file(fn).download(root=str(tmp), replace=True).name
        out.append(wandb_mod.Image(local, caption=cap))
    return out if value.get("_type") == "images/separated" else out[0]


def copy_run(src, src_entity: str, dst_entity: str, project: str, wandb_mod=None) -> list:
    """Trả về danh sách khóa không chép được (vd biểu đồ tùy biến)."""
    if wandb_mod is None:
        import wandb as wandb_mod
    new = wandb_mod.init(entity=dst_entity, project=project, id=src.id, name=src.name, group=src.group,
                         job_type=src.job_type, tags=list(src.tags or []) + ["copied"], notes=src.notes,
                         config={**dict(src.config), "copied_from": f"{src_entity}/{project}/{src.id}"},
                         resume="never")
    for row in src.scan_history():
        data = clean_row(row)
        if data:
            new.log(data, step=row.get("_step"))
    for art in src.logged_artifacts():
        if art.type == "run_table":
            key = table_key(art.name)
            t = art.get(key)
            new.log({key: wandb_mod.Table(columns=list(t.columns), data=[list(r) for r in t.data])})
    summary = src.summary._json_dict
    lost = []
    with tempfile.TemporaryDirectory() as tmp:
        for k, v in summary.items():
            if k.startswith("_") or not is_media(v) or v.get("_type") == "table-file":
                continue
            try:
                new.log({k: _images(src, v, wandb_mod, Path(tmp))})
            except Exception:
                lost.append(k)
    new.summary.update({k: v for k, v in summary.items() if not k.startswith("_") and not is_media(v)})
    new.finish()
    return lost


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src_entity", required=True)
    ap.add_argument("--dst_entity", required=True)
    ap.add_argument("--project", default="graphml-seminar-report")
    ap.add_argument("--exclude_regex", default="dryrun", help="bỏ run có tên khớp (mặc định: chạy thử)")
    ap.add_argument("--exclude_ids", nargs="*", default=[], help="bỏ các run id này (vd run hết VRAM)")
    ap.add_argument("--dry_run", action="store_true", help="chỉ in danh sách, không chép")
    a = ap.parse_args(argv)

    # thư mục run, cache artifact và file tải về: đặt dưới thư mục tạm (đặt TMP sang ổ còn trống nếu C: đầy)
    import os
    work = Path(tempfile.gettempdir()) / "wandb-copy"
    for var, sub in [("WANDB_DIR", "runs"), ("WANDB_CACHE_DIR", "cache"), ("WANDB_DATA_DIR", "data"),
                     ("WANDB_ARTIFACT_DIR", "artifacts")]:
        os.environ.setdefault(var, str(work / sub))
        Path(os.environ[var]).mkdir(parents=True, exist_ok=True)
    import wandb
    api = wandb.Api()
    src_runs = list(api.runs(f"{a.src_entity}/{a.project}"))
    try:
        existing = {r.id for r in api.runs(f"{a.dst_entity}/{a.project}")}
    except Exception:
        existing = set()
    todo, skipped = select_runs(src_runs, a.exclude_regex, set(a.exclude_ids), existing)
    for rid, name, why in skipped:
        print(f"bỏ qua {rid} {name}: {why}")
    print(f"sẽ chép {len(todo)}/{len(src_runs)} run sang {a.dst_entity}/{a.project}")
    if a.dry_run:
        for r in todo:
            print(f"  {r.id} {r.name}")
        return
    for i, r in enumerate(todo, 1):
        lost = copy_run(r, a.src_entity, a.dst_entity, a.project)
        print(f"[{i}/{len(todo)}] {r.id} {r.name}" + (f"  (không chép được: {', '.join(lost)})" if lost else ""))
    print(f"xong: https://wandb.ai/{a.dst_entity}/{a.project}")


if __name__ == "__main__":
    main()
