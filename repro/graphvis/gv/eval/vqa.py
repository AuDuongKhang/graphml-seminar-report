"""Đánh giá VQA zero-shot (Table 2, Table 6): ScienceQA-IMG, MMBench-EN dev (CircularEval), POPE (3 thiết lập).

Bài dùng script đánh giá của LLaVA (MMBench chấm trên server OpenCompass); nhóm chấm local, prompt kiểu LLaVA.
"""
import random
import re

MC_SUFFIX = "Answer with the option's letter from the given choices directly."
POPE_SUFFIX = "Answer the question using a single word or phrase."
SCIENCEQA = ("derek-thomas/ScienceQA", None, "test")
MMBENCH = ("lmms-lab-encoder/MMBench", "en", "dev")
POPE = ("lmms-lab-encoder/POPE", "Full", ("random", "popular", "adversarial"))


def _ok(v) -> bool:
    return v is not None and str(v).strip() not in ("", "nan", "None")


def scienceqa_prompt(row: dict) -> str:
    ctx = f"Context: {row['hint']}\n" if _ok(row.get("hint")) else ""
    opts = "\n".join(f"{'ABCDE'[i]}. {c}" for i, c in enumerate(row["choices"]))
    return f"{ctx}{row['question']}\n{opts}\n{MC_SUFFIX}"


def mmbench_prompt(row: dict) -> str:
    hint = f"{row['hint']}\n" if _ok(row.get("hint")) else ""
    opts = "\n".join(f"{k}. {row[k]}" for k in "ABCD" if _ok(row.get(k)))
    return f"{hint}{row['question']}\n{opts}\n{MC_SUFFIX}"


def mmbench_circular_accuracy(rows: list) -> float:
    groups = {}
    for r in rows:
        g = r["index"] % 1_000_000
        groups[g] = groups.get(g, True) and bool(r["correct"])
    return sum(groups.values()) / max(len(groups), 1)


def sample_mmbench_groups(rows: list, n_groups: int | None, seed: int = 42) -> list:
    if not n_groups:
        return rows
    ids = sorted({r["index"] % 1_000_000 for r in rows})
    keep = set(random.Random(seed).sample(ids, min(n_groups, len(ids))))
    return [r for r in rows if r["index"] % 1_000_000 in keep]


def pope_yes(text: str) -> bool:
    words = set(re.findall(r"[a-z']+", (text or "").lower()))
    return not ({"no", "not"} & words)


def pope_metrics(preds: list, golds: list) -> dict:
    p = [pope_yes(x) for x in preds]
    g = [x.strip().lower() == "yes" for x in golds]
    tp = sum(a and b for a, b in zip(p, g))
    fp = sum(a and not b for a, b in zip(p, g))
    fn = sum(not a and b for a, b in zip(p, g))
    acc = sum(a == b for a, b in zip(p, g)) / max(len(g), 1)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1, "yes_ratio": sum(p) / max(len(p), 1)}


def _load(name, config, split):
    from datasets import load_dataset
    return load_dataset(name, config, split=split) if config else load_dataset(name, split=split)


def eval_vqa(benchmark: str, adapter: str | None, exp_id: str, group: str | None = None, max_eval: int | None = None,
             batch_size: int = 8, quant: str = "4bit", dtype: str = "fp16", image_mode: str = "pad",
             seed: int = 42, model=None, proc=None) -> dict:
    """benchmark: scienceqa | mmbench | pope. max_eval: số mẫu (POPE: mỗi thiết lập; MMBench: số câu gốc)."""
    import wandb

    from repro.common.paths import out_root
    from repro.common.tracking import init_wandb

    from ..model.load import load_model_and_processor
    from .generate import generate
    from .mcqa import extract_choice_letter

    out = out_root() / "graphvis" / "eval" / exp_id
    run = init_wandb(exp_id, {"benchmark": benchmark, "adapter": adapter, "max_eval": max_eval}, out, group=group)
    if model is None:
        model, proc = load_model_and_processor(image_mode=image_mode, quant=quant, dtype=dtype,
                                               adapter_path=adapter if adapter not in (None, "none") else None)
    rng = random.Random(seed)
    summary = {}
    if benchmark == "scienceqa":
        ds = _load(*SCIENCEQA).filter(lambda r: r["image"] is not None)
        rows = [{"image": r["image"], "prompt": scienceqa_prompt(r), "answer": "ABCDE"[r["answer"]],
                 "n": len(r["choices"])} for r in ds]
        rows = rng.sample(rows, max_eval) if max_eval and max_eval < len(rows) else rows
        preds = generate(model, proc, rows, image_mode=image_mode, batch_size=batch_size, max_new_tokens=8, dtype=dtype)
        correct = [extract_choice_letter(p, r["n"]) == r["answer"] for p, r in zip(preds, rows)]
        summary["img_accuracy"] = 100 * sum(correct) / len(rows)
        run.log({"predictions": wandb.Table(columns=["gold", "raw", "correct"],
                                            data=[[r["answer"], p, c] for r, p, c in zip(rows, preds, correct)])})
    elif benchmark == "mmbench":
        ds = _load(*MMBENCH)
        rows = [{"index": r["index"], "image": r["image"], "prompt": mmbench_prompt(r), "answer": r["answer"],
                 "n": sum(_ok(r[k]) for k in "ABCD"), "category": r["category"]} for r in ds]
        rows = sample_mmbench_groups(rows, max_eval, seed)
        preds = generate(model, proc, rows, image_mode=image_mode, batch_size=batch_size, max_new_tokens=8, dtype=dtype)
        for r, p in zip(rows, preds):
            r["correct"] = extract_choice_letter(p, r["n"]) == r["answer"]
            r["raw"] = p
        summary["overall_circular"] = 100 * mmbench_circular_accuracy(rows)
        summary["single_pass"] = 100 * sum(r["correct"] for r in rows) / len(rows)
        run.log({"predictions": wandb.Table(columns=["index", "category", "gold", "raw", "correct"],
                                            data=[[r["index"], r["category"], r["answer"], r["raw"], r["correct"]]
                                                  for r in rows])})
    elif benchmark == "pope":
        name, config, splits = POPE
        for sp in splits:
            ds = _load(name, config, sp)
            rows = [{"image": r["image"], "prompt": f"{r['question']}\n{POPE_SUFFIX}", "answer": r["answer"]} for r in ds]
            rows = rng.sample(rows, max_eval) if max_eval and max_eval < len(rows) else rows
            preds = generate(model, proc, rows, image_mode=image_mode, batch_size=batch_size, max_new_tokens=4,
                             dtype=dtype)
            m = pope_metrics(preds, [r["answer"] for r in rows])
            summary.update({f"{sp}/{k}": 100 * v for k, v in m.items()})
    else:
        raise ValueError(benchmark)
    run.summary.update(summary)
    run.finish()
    return summary
