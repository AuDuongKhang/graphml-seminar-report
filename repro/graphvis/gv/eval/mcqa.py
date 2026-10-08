"""Đánh giá trắc nghiệm CSQA/OBQA với bốn chế độ: text | image | kaping | image+kaping (Table 1, Table 3)."""
import re

LETTERS = "ABCDE"
BINS = [(0, 10, "0-10"), (11, 20, "11-20"), (21, 40, "21-40"), (41, 10 ** 9, ">40")]

_LEAD_UPPER = re.compile(r"^\(?([A-E])\)?(?:[\s.):]|$)")
_LEAD_LOWER = re.compile(r"^\(?([a-e])[.)]")
_ANSWER_IS = re.compile(r"answer\s*(?:is|:)?\s*:?\s*\(?([A-E])\)?(?:\b|$)", re.IGNORECASE)


def extract_choice_letter(text: str, n_choices: int):
    t = (text or "").strip()
    m = _LEAD_UPPER.match(t) or _LEAD_LOWER.match(t)
    if not m:
        m = _ANSWER_IS.search(t)
    if not m:
        return None
    letter = m.group(1).upper()
    return letter if LETTERS.index(letter) < n_choices else None


def node_bin(n: int) -> str:
    for lo, hi, name in BINS:
        if lo <= n <= hi:
            return name
    raise ValueError(n)


def accuracy_by_bins(n_nodes: list, correct: list) -> dict:
    tot, ok = {}, {}
    for n, c in zip(n_nodes, correct):
        b = node_bin(n)
        tot[b] = tot.get(b, 0) + 1
        ok[b] = ok.get(b, 0) + int(c)
    return {b: ok[b] / tot[b] for _, _, b in BINS if b in tot}


def insert_kaping(prompt: str, prefix: str) -> str:
    """Chèn khối bộ ba KAPING ngay trước 'Question:' (giữ nguyên phần mô tả ảnh nếu có)."""
    if not prefix:
        return prompt
    i = prompt.find("Question:")
    return prefix + prompt if i <= 0 else prompt[:i] + prefix + prompt[i:]


def n_choices_in(prompt: str) -> int:
    return len(re.findall(r"^[A-E]\. ", prompt, flags=re.MULTILINE))


def eval_mcqa(dataset: str, split: str, mode: str, adapter: str | None, exp_id: str, group: str | None = None,
              data_tag: str = "", max_eval: int | None = None, batch_size: int = 8, quant: str = "4bit",
              dtype: str = "fp16", image_mode: str = "pad", kaping_k: int = 10,
              image_size: int | None = None, model=None, proc=None) -> dict:
    import json

    import wandb

    from repro.common.paths import out_root
    from repro.common.tracking import init_wandb

    from ..data.build import read_rows
    from ..data.kaping import kaping_prefix, top_k_triples
    from ..model.load import load_model_and_processor
    from ..pipeline import load_graphs, split_dir
    from .generate import generate

    d = split_dir(dataset, split, data_tag)
    src = "dv.jsonl" if mode.startswith("image") else "text.jsonl"
    rows = read_rows(d / src, limit=max_eval)
    if "kaping" in mode:
        graphs = load_graphs(d / "graphs.jsonl")
        for r in rows:
            q = r.get("question") or r["prompt"].split("Question:", 1)[-1].split("\n", 1)[0]
            r["prompt"] = insert_kaping(r["prompt"], kaping_prefix(top_k_triples(q, graphs[r["id"]].triples, k=kaping_k)))
    out = out_root() / "graphvis" / "eval" / exp_id
    run = init_wandb(exp_id, {"dataset": dataset, "split": split, "mode": mode, "adapter": adapter,
                              "n": len(rows), "data_tag": data_tag, "quant": quant, "image_size": image_size,
                              "image_mode": image_mode}, out, group=group)
    if model is None:
        model, proc = load_model_and_processor(image_mode=image_mode, quant=quant, dtype=dtype,
                                               adapter_path=adapter if adapter not in (None, "none") else None)
    preds = generate(model, proc, rows, image_mode=image_mode, batch_size=batch_size, max_new_tokens=8, dtype=dtype,
                     image_size=image_size)
    letters = [extract_choice_letter(p, n_choices_in(r["prompt"])) for p, r in zip(preds, rows)]
    correct = [l == r["answer"] for l, r in zip(letters, rows)]
    acc = sum(correct) / max(len(rows), 1)
    bins = accuracy_by_bins([r["n_nodes"] for r in rows], correct)
    table = wandb.Table(columns=["id", "n_nodes", "gold", "pred_letter", "raw"],
                        data=[[r["id"], r["n_nodes"], r["answer"], l, p] for r, l, p in zip(rows, letters, preds)])
    run.log({"predictions": table, **{f"accuracy_by_nodes/{k}": v for k, v in bins.items()}})
    run.summary.update({"accuracy": acc, "unparsed": sum(l is None for l in letters)})
    (out / "predictions.jsonl").write_text("\n".join(json.dumps({"id": r["id"], "n_nodes": r["n_nodes"], "gold": r["answer"],
                                                                 "pred": l, "raw": p}) for r, l, p in zip(rows, letters, preds)))
    run.finish()
    return {"accuracy": acc, "by_nodes": bins}
