"""Ghi hai lượt W&B ngoại tuyến; đăng công khai chỉ khi người dùng cung cấp dự án."""
import argparse
import json
from pathlib import Path
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--entity")
    parser.add_argument("--project", default="scimkg-pilot")
    parser.add_argument("--online", action="store_true", help="Cần đăng nhập và dự án công khai đã cấu hình")
    args = parser.parse_args()
    if args.online and not args.entity: parser.error("--online cần --entity đã xác định")
    os.environ["WANDB_MODE"] = "online" if args.online else "offline"
    args.directory.mkdir(parents=True, exist_ok=True)
    import wandb
    for folder in args.run:
        summary = json.loads((folder / "summary.json").read_text())
        audit = json.loads((folder / "audit.json").read_text())
        config = json.loads((folder / "config.json").read_text())
        with wandb.init(project=args.project, entity=args.entity, mode="online" if args.online else "offline",
                        name=summary["lesson_id"], dir=str(args.directory.resolve()), config=config,
                        tags=["pilot", "subscription", "provisional-labels", "awaiting-human-review"]) as run:
            run.log({"graph/nodes": summary["graph"]["nodes"], "graph/edges": summary["graph"]["edges"],
                     "usage/actual_calls": audit["usage"]["actual_calls"], "usage/input_tokens": audit["usage"]["input_tokens"],
                     "usage/output_tokens": audit["usage"]["output_tokens"],
                     "usage/inference_seconds": audit["usage"]["elapsed_inference_seconds"],
                     **{"provisional_metrics/" + name: value for name, value in summary["extraction_metrics"].items() if name in ["precision", "recall", "f1"]}})
            run.summary["annotation_provenance"] = summary["extraction_metrics"]["annotation_provenance"]
            run.summary["human_review"] = "pending"
            run.summary["api_cost_measurement"] = "not_measured_subscription"
            run.summary["missing_video"] = summary["missing_video"]
            # Chỉ số và cấu hình; không đẩy prompt, dữ liệu gốc hoặc sự kiện đăng nhập lên W&B.
            identifier = run.id
        receipt = {"id": identifier, "mode": "online" if args.online else "offline", "project": args.project,
                   "entity": args.entity, "public_url": None, "human_review": "pending"}
        (folder / "wandb-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
        print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__": main()
