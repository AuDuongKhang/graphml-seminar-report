"""Baseline của Table 4, cài lại theo mô tả trong các bài gốc (mã không đi kèm SciMKG).

Mỗi baseline: run_<tên>(lesson, client, subject) -> {mốc thời gian: [khái niệm]}. LinkNER không cài (cần
một mô hình NER cục bộ đã huấn luyện cho miền này — không có dữ liệu huấn luyện).
"""
import json
import re
from collections import Counter

from .llm import parse_concepts
from .normalize import normalize_concept

_SPAN = re.compile(r"@@(.+?)##")


def _norm_list(items):
    out, seen = [], set()
    for x in items:
        c = normalize_concept(x)
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def extract_gptner_spans(text: str) -> list:
    return _norm_list(_SPAN.findall(text or ""))


def parse_json_list(text: str) -> list:
    m = re.search(r"\[.*?\]", text or "", re.DOTALL)
    if not m:
        return []
    try:
        val = json.loads(m.group())
    except json.JSONDecodeError:
        return []
    return _norm_list(str(x) for x in val if isinstance(x, (str, int, float)))


def majority_vote(samples: list) -> list:
    """Giữ phần tử xuất hiện trong hơn một nửa số lần lấy mẫu (self-consistency)."""
    if not samples:
        return []
    c = Counter(x for s in samples for x in set(s))
    order = [x for s in samples for x in s]
    return [x for x in dict.fromkeys(order) if c[x] > len(samples) / 2]


def _ngrams(s: str, n: int = 2) -> Counter:
    s = s or ""
    return Counter(s[i:i + n] for i in range(max(len(s) - n + 1, 1)))


def top_k_similar(query: str, pool: list, k: int = 4) -> list:
    """Truy hồi ví dụ minh họa theo cosine trên bigram ký tự (đủ cho văn bản tiếng Trung)."""
    q = _ngrams(query)

    def cos(t):
        v = _ngrams(t)
        dot = sum(q[g] * v[g] for g in q)
        nq = sum(x * x for x in q.values()) ** 0.5
        nv = sum(x * x for x in v.values()) ** 0.5
        return dot / (nq * nv) if nq and nv else 0.0
    return sorted(pool, key=cos, reverse=True)[:k]


# ---------------- Decomposed-QA (Xie et al., 2023): tách cú pháp rồi hỏi theo loại ----------------
DQA_STEP1 = ("Text: {text}\nList all nouns and noun phrases in the text, one per line starting with '-'. "
             "Output only the list.")
DQA_STEP2 = ("Text: {text}\nCandidate noun phrases:\n{nps}\nQuestion: Which of these candidates are {subject} concepts "
             "(K-12 {subject} terminology)? Answer in Simplified Chinese as a list, one per line starting with '-'. "
             "Output only the list.")


def run_decomposed_qa(lesson, client, subject: str) -> dict:
    out = {}
    for t, text in lesson.segments:
        nps, _ = parse_concepts(client.chat(DQA_STEP1.format(text=text)).text)
        if not nps:
            out[t] = []
            continue
        items, _ = parse_concepts(client.chat(DQA_STEP2.format(text=text, nps="\n".join(f"- {x}" for x in nps),
                                                               subject=subject)).text)
        out[t] = _norm_list(items)
    return out


# ---------------- GPT-NER (Wang et al., 2023): đánh dấu @@…## rồi tự kiểm chứng ----------------
GPTNER_MARK = ("I am an excellent linguist. The task is to label {subject} concepts in the given text. "
               "Rewrite the text exactly, surrounding every {subject} concept with @@ and ##, "
               "e.g. @@细胞膜## is a {subject} concept.\nInput: {text}\nOutput:")
GPTNER_VERIFY = ('The task is to verify whether the word is a {subject} concept extracted from the given text.\n'
                 'Text: {text}\nWord: "{word}"\nIs the word a {subject} concept in the text? Answer yes or no.')


def run_gpt_ner(lesson, client, subject: str, verify: bool = True) -> dict:
    out = {}
    for t, text in lesson.segments:
        spans = extract_gptner_spans(client.chat(GPTNER_MARK.format(subject=subject, text=text)).text)
        if verify:
            spans = [w for w in spans
                     if (client.chat(GPTNER_VERIFY.format(subject=subject, text=text, word=w)).text or "")
                     .strip().lower().startswith("yes")]
        out[t] = spans
    return out


# ---------------- Self-Improving (Xie et al., 2024): tự gán nhãn + self-consistency + ví dụ truy hồi ----------------
SI_ZERO = ("Extract all {subject} concepts from the text. Answer in Simplified Chinese as a list, one per line "
           "starting with '-'. Output only the list.\nText: {text}")
SI_FEW = ("Extract all {subject} concepts from the text. Answer in Simplified Chinese as a list, one per line "
          "starting with '-'. Output only the list.\n{demos}\nText: {text}")


def run_self_improving(lesson, client, subject: str, corpus_segments: list, n_samples: int = 5, k: int = 4) -> dict:
    """corpus_segments: các đoạn không nhãn cùng môn (từ những bài khác) dùng để tự gán nhãn."""
    labeled = {}
    for text in corpus_segments:
        samples = [parse_concepts(client.chat(SI_ZERO.format(subject=subject, text=text), temperature=0.7,
                                              seed=42 + i).text)[0] for i in range(n_samples)]
        voted = majority_vote([_norm_list(s) for s in samples])
        if voted:
            labeled[text] = voted
    out = {}
    for t, text in lesson.segments:
        demos = "\n".join(f"Text: {d}\nConcepts:\n" + "\n".join(f"- {c}" for c in labeled[d]) + "\n"
                          for d in top_k_similar(text, list(labeled), k))
        out[t] = _norm_list(parse_concepts(client.chat(SI_FEW.format(subject=subject, demos=demos, text=text)).text)[0])
    return out


# ---------------- UniversalNER (Zhou et al., 2023): UniNER-7B-type qua vLLM (chế độ completion) ----------------
def uniner_prompt(text: str, label: str) -> str:
    return ("A virtual assistant answers questions from a user based on the provided text.\n"
            f"USER: Text: {text}\nASSISTANT: I've read this text.\n"
            f"USER: What describes {label} in the text?\nASSISTANT:")


def run_universal_ner(lesson, client, subject: str) -> dict:
    return {t: parse_json_list(client.chat(uniner_prompt(text, f"{subject} concept")).text)
            for t, text in lesson.segments}


BASELINES = {"decomposed_qa": run_decomposed_qa, "gpt_ner": run_gpt_ner, "self_improving": run_self_improving,
             "universal_ner": run_universal_ner}
