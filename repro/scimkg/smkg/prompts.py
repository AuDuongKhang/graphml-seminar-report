"""Prompt của SciMKG, chép nguyên văn từ mã tác giả (SciMKG/ExtractMultimodalConcepts/NER.py,
SciMKG/MultimodalAlignment/ImageAligment.py). Chỗ điền: {subject}, {candidates}, {feedback}, {concepts}.

Cách ghép giống mã gốc: EXTRACT được nối SAU văn bản đoạn; FEEDBACK được nối SAU danh sách ứng viên và
SAU đó là phần phản hồi tích lũy ("hello, " + các phản hồi trước).
"""

EXTRACT = """
            $
            Silently analyze the text to:
            a) Identify all nouns/noun phrases using syntactic tagging
            b) Filter these to retain only concepts related to {subject}
            c) STRICTLY OUTPUT ONLY the final {subject} concepts: In Simplified Chinese
            As a single list using this exact format: -xxx\n-xxx\n-xxx
            No headers/explanations/other text
            Never reveal non-subject nouns or intermediate analysis.
            $
            """

FEEDBACK = """
    $
    ---
    Analyze the extracted {subject} concepts to:
    a) NEVER modify or split original terms
    b) Identify terms with cross-disciplinary ambiguity (terms that hold distinct meanings in other academic fields)
    c) For each ambiguous term, provide a one-sentence clarification of its potential alternative academic meaning
    d) Format output EXACTLY as: "ambiguous term" ："reason"
    e) Output ONLY lines following this format without additional text
    f) Preserve original lexical structure

    Example analysis pattern:
    "系统" : "Could refer to computer systems in technology contexts"

    $
    """

REFINE = """
                $
                Concept Filtering Protocol for {subject}

                FEEDBACK ANALYSIS
                $
                {feedback}
                $
                (Note: Feedback analysis may contain errors.)
                (Each feedback item clearly indicates terms that are out-of-scope or ambiguous with respect to  {subject}.

                CANDIDATE POOL
                "{candidates}"

                1.Perform primary pruning using the FEEDBACK ANALYSIS to remove terms that are not directly relevant to  {subject}.
                2.Secondary filtration: Apply  {subject} standards to ensure that all remaining terms are inherently  {subject} concepts rather than instruments, apparatuses, or tools merely utilized in {subject} experiments.
                3. Eliminate redundancies and non-{subject} terms while preserving the original order.

                OUTPUT SPECIFICATIONS
                Strictly output in Markdown list format, with each term on a new line as follows:
                - concept1
                - concept2
                - concept3
                Maintain the original order of valid entries.
                Output only the list, with no additional text.

                Filtered Result:
                $

                """

IMAGE_ALIGN = """
        You are an AI  {subject} image analyst. Follow this strict protocol:
        1. Candidate Verification:
           Within the {subject} concepts {concepts},
           identify and extract the concepts that appear in the image or are highly related to it.
           Use these extracted concepts to form the C4 .
        2. Multilevel Analysis:
        a) OCR Extraction: Capture ALL {subject} pure concept  in  → C1
        b) Regional Analysis: Identify specific  {subject} components → C2
        c) Holistic Analysis: Derive tissue/system-level {subject} concepts → C3
        3. Deduplication Protocol:
        Priority: C4 > C3 > C2 > C1
        - Remove duplicates ONLY for identical terms
        - Preserve hierarchical terms (e.g., keep both "上皮组织" and "小肠上皮细胞")
        4. Output Decision:
        - If any valid terms remain: List ALL in bullet format
        - If completely empty: Output "none"
        Mandatory Format:
        - concept1
        - concept2
        ...

        OR
        "none"
        Absolute constraints:
        1. No term suppression except exact duplicates
        2. No explanations, categories, or formatting variations.
    """

SUBJECT_NAME = {"biology": "Biology", "physics": "Physics", "chemistry": "Chemistry"}


def fill(template: str, **kw) -> str:
    out = template
    for k, v in kw.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def extract_prompt(segment_text: str, subject: str) -> str:
    return segment_text + fill(EXTRACT, subject=subject)


def feedback_prompt(candidates: list, subject: str, history: str) -> str:
    return str(candidates) + fill(FEEDBACK, subject=subject) + history


def refine_prompt(candidates: list, subject: str, feedback: str) -> str:
    return fill(REFINE, subject=subject, feedback=feedback, candidates=candidates)
