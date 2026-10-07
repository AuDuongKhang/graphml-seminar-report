# Dựng sẵn toàn bộ thực nghiệm tái lập GraphVis + SciMKG — Kế hoạch triển khai

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Người dùng mở 2 notebook Colab, điền khóa API/W&B, chạy từng ô theo ID thực nghiệm (G0–G7, C1–C3, S0–S5) và nhận run W&B công khai — không phải viết thêm code.

**Architecture:** Mỗi bài một package Python nhỏ trong `repro/` (`gv` cho GraphVis, `smkg` cho SciMKG), dùng chung `repro/common` để khởi tạo W&B. Mọi thực nghiệm được gọi qua một CLI duy nhất mỗi bài, cấu hình bằng YAML theo ID thực nghiệm. Phần chạy được trên CPU (dựng dữ liệu, chấm điểm, prompt, metric) có test pytest chạy local; phần cần GPU/API chỉ chạy trên Colab, có chế độ `--dry_run` (vài mẫu) để kiểm tra nhanh.

**Tech Stack:** Python 3.10+, PyTorch, HF `transformers` (`LlavaNextForConditionalGeneration`, checkpoint `llava-hf/llava-v1.6-mistral-7b-hf`), `peft`, `bitsandbytes`, `datasets`, `graphviz`, `networkx`, `sentence-transformers` (KAPING), `wandb`, `openai` (client OpenAI-compatible cho GPT-4o/DeepSeek/Gemini/Ollama/vLLM), `open_clip`/`transformers` CLIP & BLIP, `pytest`.

**Spec:** `docs/experiments.md` (danh sách ID), `docs/findings.md` (số liệu bài + điểm không khớp bài↔mã), `docs/context.md` (yêu cầu môn học: W&B công khai theo từng mục, README chỉ hướng dẫn chạy).

## Global Constraints

- Mỗi thực nghiệm = 1 W&B run (hoặc 1 group) tên `<ID>-<mô tả>`; config log đủ: siêu tham số, seed, `git rev-parse HEAD`, ID thực nghiệm.
- Không commit dữ liệu, ảnh sinh ra, checkpoint, khóa API. Dữ liệu/ckpt trên Colab lưu ở Google Drive `/content/drive/MyDrive/graphml-repro/`.
- README chỉ ghi cách chạy (môi trường, phần cứng, lệnh, notebook); không ghi kết quả.
- Mọi sai khác so với bài báo phải có cờ cấu hình tường minh + ghi vào `repro/*/DEVIATIONS.md` (nguồn cho mục VII-A/XII-A của báo cáo).
- Seed mặc định 42; giải mã tham lam (`do_sample=False`) cho mọi đánh giá GraphVis; `temperature=0, top_p=0.1, frequency_penalty=0.5, seed=42, max_tokens=1000` cho SciMKG (Bảng 6 bài).
- Code identifiers tiếng Anh; docstring/README tiếng Việt ngắn gọn.
- Local Windows chỉ có CPU, `transformers` cũ, không có `dot` → test local không được import `transformers`/`torch` GPU; test cần `dot` dùng `pytest.mark.skipif(shutil.which("dot") is None)`.

## Quyết định thiết kế (đã cân nhắc, ghi vào DEVIATIONS.md)

| # | Vấn đề | Quyết định | Lý do |
|---|---|---|---|
| D1 | Mã LLaVA của tác giả (fork LLaVA cũ, đọc trường `chosen`, nạp Mistral bằng lớp Llama, cần deepspeed/flash-attn) | Viết lại vòng huấn luyện gọn bằng HF `transformers` + `peft` trên `llava-hf/llava-v1.6-mistral-7b-hf`; tái hiện đúng: LoRA r128/α256 trên mọi `nn.Linear` của LLM, projector học toàn bộ lr 2e-5, vision tower đóng băng, loss chỉ trên token trả lời, 1 epoch/giai đoạn, cosine, warmup 0.03, wd 0, max_len 2048 | Chạy được trên Colab hiện đại; code của tác giả không chạy được nguyên trạng (findings §A3) |
| D2 | `image_aspect_ratio=pad` (1 ô 336 → 576 token) không có trong HF LlavaNext | `image_mode=pad`: tự đệm ảnh thành vuông bằng màu trung bình CLIP, đặt `image_grid_pinpoints=[[336,336]]` cho processor+model (= ảnh gốc 336 + 1 ô 336, 1152 token, cùng thông tin). Có thêm `image_mode=anyres` (mặc định HF) | Gần nhất với `pad` mà vẫn dùng kiến trúc chuẩn |
| D3 | lr bài 1e-7 vs script 2e-5; batch 4 vs 32 | Mặc định lr **2e-5**, batch toàn cục **4** (per_device 1–4 + grad accumulation). Cờ `--lr` để thử 1e-7 | 1e-7 với 1 epoch gần như không cập nhật LoRA; batch 4 theo bài |
| D4 | Truy hồi đồ thị con: tác giả dùng file `*.graph.jsonl` của MHGRN (không công bố) | Tự truy hồi từ `grounded.jsonl` + `conceptnet.en.pruned.graph` của QA-GNN: nút = qc ∪ ac ∪ {x : x kề cả q∈qc và a∈ac} (đường đi ≤2 bước), cạnh = mọi cạnh ConceptNet giữa các nút đó; gộp mọi lựa chọn của một câu. Cờ `--max_nodes` (mặc định 80, giữ qc/ac, cắt nút trung gian theo số đường đi qua nó) | Đúng định nghĩa "mọi đường đi k-bước nối các khái niệm" của bài; log phân bố số nút/cạnh lên W&B để so với Fig 4–5 |
| D5 | CSQA test không có nhãn | Train trên official train (9.741), đánh giá trên official dev (1.221). OBQA: train 4.957 / test 500 | Bài không nói split; đây là lựa chọn phổ biến |
| D6 | Đánh giá VQA của LLaVA (MMBench nộp lên server) | Tự chấm local: ScienceQA-IMG test (2.017), MMBench-EN dev (4.329 dòng, CircularEval theo `index % 1_000_000`), POPE 3×3.000 (Acc, F1) | Không cần server; ghi rõ khác biệt |
| D7 | "Base LVLM" trong Table 1 không rõ có ảnh hay không | Chạy cả `base-text` và `base-image`, báo cả hai | Minh bạch |
| D8 | SciMKG Integration bug (findings §B2) | Ghi lại phiếu theo từng (LLM, vòng); tích hợp offline với tỉ lệ `votes/(|M|·K)`. Table 5 (iter × α) và w/o MLs tính lại từ cùng log, không gọi API thêm | Đúng ý bài, tiết kiệm chi phí |
| D9 | SciMKG metric không được định nghĩa | P/R/F1 micro trên tập khái niệm theo bài; ACC/MCC trên vũ trụ `U_l = GT_l ∪ pool_l` (pool = mọi ứng viên mọi phương pháp sinh ra cho bài l); AUC dùng điểm tỉ lệ phiếu (baseline nhị phân → AUC = balanced accuracy). So khớp chuỗi sau chuẩn hóa (NFKC, bỏ khoảng trắng/dấu câu) | Ghi rõ trong báo cáo |
| D10 | Mô hình API đã ngừng (Gemini-1.5) | `configs/models.yaml` ánh xạ tên-bài → model thực tế; script `check_models` thử gọi từng model trước khi chạy và in bảng thay thế | Theo quyết định: đúng cấu hình nếu còn, không thì ghi rõ |
| D11 | Baseline LinkNER cần mô hình NER cục bộ đã huấn luyện trên miền này (không có dữ liệu train) | Cài 4 baseline: Decomposed-QA, GPT-NER, Self-Improving, UniversalNER (UniNER-7B-type, chạy local). LinkNER: **bỏ, ghi lý do** | Không tái hiện trung thực được |
| D12 | Table 7 cần X-VLM (không có bản HF) và VQAScore (CLIP-FlanT5 11B) | CLIPScore + BLIP-ITM (HF) bắt buộc; VQAScore qua `t2v_metrics` bản `clip-flant5-xl` (tùy chọn, A100); X-VLM tùy chọn qua repo gốc; "Human" = 2 thành viên chấm 100 cặp qua file CSV sinh sẵn | Khả thi trên Colab |

## Ràng buộc tài nguyên đã chốt (07/10/2026): chỉ nền tảng miễn phí, chưa có API key

Áp dụng ưu tiên trên mọi chỗ khác trong kế hoạch:

- **Nền tảng mục tiêu:** Kaggle Notebooks (2×T4 16 GB hoặc P100, ~30 giờ GPU/tuần, phiên 12 giờ, chạy nền) là chính; Colab miễn phí (T4) là phụ. **Không có bf16** trên T4/P100 → mặc định `quant: 4bit`, `dtype: fp16` (QLoRA). Notebook phát hiện nền tảng (`/kaggle/working` hay Colab+Drive) để đặt `GRAPHML_DATA`/`GRAPHML_OUT`. Checkpoint/adapter có thể đẩy lên HuggingFace Hub (miễn phí, repo private) qua cờ `--hub_repo` để chuyển giữa Kaggle và Colab.
- **Preset quy mô** (`--scale`): `tiny` (smoke test, ~200 mẫu/giai đoạn, 100 mẫu eval), `lite` (**mặc định**: D_g 600 ảnh × 6 = 3,6k; D_v 2.000; eval CSQA dev đủ 1.221, OBQA test đủ 500; VQA lấy mẫu cố định seed: ScienceQA-IMG đủ 2.017, MMBench-EN dev đủ, POPE 500/split), `full` (như bài). Ước lượng `lite` trên T4 QLoRA: ~5–7 giờ/lượt curriculum → 7 lượt ≈ 40–50 giờ GPU (~2 tuần quota Kaggle). Cờ `--max_eval` cho mọi lệnh eval.
- **SciMKG không có key:** mặc định "cấu hình bài" được thay bằng bộ ba mô hình mở lớn nhất của chính Bảng 4 trong bài — **Qwen3-14B + Gemma3-27B + Phi-3-medium** qua Ollama (Kaggle 2×T4 = 32 GB đủ cho Gemma3-27B q4). Phi-3-small (7B, không có trên Ollama) trong bộ cỡ vừa được thay bằng `phi3:mini`/ghi chú, hoặc chạy vLLM nếu GPU đủ. Ghi rõ trong DEVIATIONS và báo cáo. `models.yaml` vẫn giữ các mục GPT-4o/DeepSeek/Gemini (OpenAI-compatible) để chạy được ngay nếu sau này có key hoặc dùng gói miễn phí có giới hạn (Gemini API free tier, GitHub Models, OpenRouter `:free`) — chỉ cho tập con nhỏ.
- **Thông lượng SciMKG:** `LLMClient` hỗ trợ gọi song song (`--workers`, đặt `OLLAMA_NUM_PARALLEL`); mặc định `--lessons 30` lấy phân tầng theo môn (seed 42), `--lessons all` cho 99 bài.

## Ước lượng chi phí (để chọn `--scale`)

| Hạng mục | full (giống bài) | lite (mặc định đề xuất nếu GPU yếu) |
|---|---|---|
| D_g CSQA | 9.741 ảnh × 6 = 58.4k mẫu | 2.000 ảnh × 6 = 12k |
| D_v CSQA | 9.741 | 9.741 (giữ đủ — QA là phần chính) |
| 1 lượt huấn luyện curriculum, A100 40GB, LoRA bf16 | ~10–12 giờ | ~4 giờ |
| Số lượt huấn luyện | curriculum, joint, graph-first, OCR-first, low-res 50×50, text-only (C3), OBQA-curriculum = 7 | 7 |
| Đánh giá VQA (1 mô hình) | ~1,5 giờ | ~1,5 giờ |
| SciMKG EVIA đầy đủ (99 bài, ~1,5k đoạn, 3 LLM × 5 vòng × 2 lời gọi) | ~45k lời gọi, ước $50–100 API | giới hạn `--lessons 30` |
| SciMKG mô hình mở (Qwen3/Gemma3/Phi-3, 3 kích cỡ) | Colab A100, ~3–5 giờ/bộ | – |

Mọi lượt huấn luyện lưu checkpoint mỗi 500 bước lên Drive và **resume** được (`--resume`), vì phiên Colab bị ngắt.

## Review Focus

1. **Phiên Colab bị ngắt giữa chừng** → chạy lại cùng lệnh phải tiếp tục từ checkpoint/cache, không làm lại từ đầu, không tạo W&B run trùng (dùng `wandb.init(id=..., resume="allow")` với id lưu trong thư mục output). Test: `test_run_id_is_stable_across_restarts` (Task 1).
2. **LLM trả về rác/định dạng sai/rỗng/timeout** → không crash; ghi vào bộ đếm lỗi FE/IO/NR và cache; retry có giới hạn. Test: `test_parse_concepts_handles_garbage`, `test_client_counts_no_response` (Task 11).
3. **Đồ thị con rỗng hoặc khổng lồ** (không có qc/ac khớp, hoặc hàng trăm nút trung gian) → rỗng: mẫu vẫn tạo được ảnh "không có tri thức" và vẫn có trong D_v (bài nói ảnh "có thể không hữu ích"); khổng lồ: cắt theo `--max_nodes`. Test: `test_empty_grounding_yields_placeholder_graph`, `test_max_nodes_keeps_question_and_answer_concepts` (Task 3).
4. **Chuỗi tiếng Trung khác nhau về dạng full-width/khoảng trắng** giữa dự đoán và nhãn vàng → chuẩn hóa trước khi so khớp. Test: `test_normalize_fullwidth_and_spaces` (Task 10).
5. **Đáp án mô hình không đúng định dạng chữ cái** ("The answer is (B).", "b", "B. bank") → trích được chữ cái; không trích được thì tính sai, không crash. Test: `test_extract_choice_letter_variants` (Task 6).

---

## Cấu trúc file

```
AGENTS.md                              # trỏ sang CLAUDE.md (cho Codex/Cursor/agent khác)
repro/
  common/
    __init__.py
    tracking.py                        # init_wandb(exp_id, config, out_dir) -> run; git hash; run id ổn định
    paths.py                           # DATA_ROOT/OUT_ROOT (env GRAPHML_DATA, GRAPHML_OUT), mặc định ./_data ./_out
  graphvis/
    README.md                          # cách chạy (không kết quả)
    DEVIATIONS.md                      # D1–D7 chi tiết
    requirements.txt
    configs/                           # 1 YAML / thực nghiệm
      G1_csqa_curriculum.yaml  G1_obqa_curriculum.yaml  G4_joint.yaml  G4_graph_first.yaml
      G4_ocr_first.yaml  G7_lowres.yaml  C3_text_only.yaml
    gv/
      __init__.py
      cli.py                           # python -m gv.cli <command> --config ...
      data/download.py                 # QA-GNN selective unzip + HF datasets
      data/remote_zip.py               # đọc zip qua HTTP Range (dùng chung với smkg qua copy nhỏ? -> đặt ở repro/common)
      data/retrieve.py                 # build_question_graph(...)
      data/render.py                   # render_graph(...)-> PNG
      data/tasks.py                    # 6 tác vụ D_g, nhóm OCR/GRAPH, prompt pools
      data/build.py                    # build_dg / build_dv / build_text_only -> jsonl
      data/kaping.py                   # top_k_triples(...)
      model/load.py                    # load_model_and_processor(...)
      model/collate.py                 # build_chat, mask labels
      train.py                         # run_stages(config)
      eval/mcqa.py                     # CSQA/OBQA: modes text|image|kaping|image+kaping
      eval/graph_reading.py            # 6 tác vụ, metric Table 4
      eval/vqa.py                      # scienceqa_img, mmbench_circular, pope
      eval/complexity.py               # C1 bins, C2 resolution sweep
      eval/generate.py                 # batched greedy generate
    tests/ (test_retrieve.py, test_tasks.py, test_collate.py, test_metrics.py, test_kaping.py)
    notebooks/GraphVis_Colab.ipynb
  scimkg/
    README.md  DEVIATIONS.md  requirements.txt
    configs/models.yaml                # paper_name -> provider, model, base_url env
    smkg/
      __init__.py
      cli.py
      data.py                          # download_subset(), load_lessons(), load_groundtruth()
      normalize.py                     # normalize_concept()
      llm.py                           # LLMClient: OpenAI-compatible, disk cache, error counters
      prompts.py                       # EXTRACT, FEEDBACK, REFINE, IMAGE_ALIGN (theo mã tác giả), baseline prompts
      evia.py                          # run_extraction, run_validation (log phiếu), integrate(votes, K, models, alpha)
      augment.py                       # ConceptNet + Wikipedia (tùy chọn, không có trong đánh giá Table 4)
      baselines.py                     # decomposed_qa, gpt_ner, self_improving, universal_ner
      metrics.py                       # set_prf, universe_metrics(acc, mcc, auc)
      errors.py                        # classify_errors -> FE/IO/CE/CO/NR
      align_eval.py                    # clip_score, blip_itm, vqascore(optional), human sheet
    tests/ (test_normalize.py, test_evia.py, test_metrics.py, test_llm.py, test_data.py)
    notebooks/SciMKG_Colab.ipynb
```

---

## Phần 0 — Nền chung

### Task 0: AGENTS.md + khung `repro/common`

**Files:**
- Create: `AGENTS.md`, `repro/common/__init__.py`, `repro/common/tracking.py`, `repro/common/paths.py`, `repro/common/remote_zip.py`, `repro/common/tests/test_tracking.py`, `repro/pytest.ini`

**Interfaces:**
- Produces:
  - `paths.data_root() -> pathlib.Path`, `paths.out_root() -> pathlib.Path` (env `GRAPHML_DATA`, `GRAPHML_OUT`).
  - `tracking.init_wandb(exp_id: str, config: dict, out_dir: Path, project: str | None = None, mode: str | None = None) -> wandb.Run` — đọc/ghi `out_dir/wandb_run_id.txt` để resume; thêm `config["git_commit"]`, `config["exp_id"]`; `project` mặc định env `WANDB_PROJECT` hoặc `"graphml-seminar-report"`; `mode` mặc định env `WANDB_MODE`.
  - `tracking.stable_run_id(out_dir: Path) -> str`.
  - `remote_zip.open_remote_zip(url: str) -> zipfile.ZipFile`; `remote_zip.extract_members(url, predicate, dest) -> list[Path]`; `remote_zip.decoded_name(info) -> str` (GBK fallback).

- [ ] **Step 1: AGENTS.md**

```markdown
# AGENTS.md
Hướng dẫn cho mọi coding agent (Codex, Cursor, Gemini CLI, ...). Nội dung chuẩn nằm ở `CLAUDE.md` —
đọc file đó trước, sau đó `docs/context.md`, `docs/findings.md`, `docs/experiments.md`.
Code tái lập ở `repro/` (xem `repro/graphvis/README.md`, `repro/scimkg/README.md`).
```

- [ ] **Step 2: Test tracking (thất bại trước)**

```python
# repro/common/tests/test_tracking.py
from repro.common.tracking import stable_run_id

def test_run_id_is_stable_across_restarts(tmp_path):
    a = stable_run_id(tmp_path)
    b = stable_run_id(tmp_path)
    assert a == b and len(a) >= 8
    assert (tmp_path / "wandb_run_id.txt").read_text().strip() == a
```

Run: `cd repro && python -m pytest common/tests -q` → FAIL (module chưa có).

- [ ] **Step 3: Cài `paths.py`, `tracking.py`, `remote_zip.py`** (remote_zip = bản sạch của `HttpRangeFile` đã dùng khi khảo sát Zenodo/QA-GNN: HEAD lấy `Content-Length`, `read()` gửi `Range: bytes=a-b`, bọc `io.BufferedReader(buffer_size=1<<16)`; `decoded_name`: nếu `flag_bits & 0x800` dùng tên gốc, ngược lại thử `cp437→gbk`).

```python
# repro/common/tracking.py (lõi)
import os, subprocess, uuid
from pathlib import Path

def stable_run_id(out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / "wandb_run_id.txt"
    if f.exists():
        return f.read_text().strip()
    rid = uuid.uuid4().hex[:12]
    f.write_text(rid)
    return rid

def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"

def init_wandb(exp_id, config, out_dir, project=None, mode=None):
    import wandb
    cfg = dict(config, exp_id=exp_id, git_commit=git_commit())
    return wandb.init(project=project or os.environ.get("WANDB_PROJECT", "graphml-seminar-report"),
                      name=exp_id, id=stable_run_id(Path(out_dir)), resume="allow",
                      config=cfg, mode=mode or os.environ.get("WANDB_MODE"), dir=str(out_dir))
```

- [ ] **Step 4: Chạy test** → PASS. `repro/pytest.ini`: `[pytest]\ntestpaths = common/tests graphvis/tests scimkg/tests\npythonpath = . graphvis scimkg ..`
- [ ] **Step 5: Commit** `feat(repro): add shared tracking, paths, remote zip; AGENTS.md`

---

## Phần A — GraphVis

### Task 1: Tải dữ liệu GraphVis (G0, phần tải)

**Files:** Create `repro/graphvis/gv/data/download.py`, `repro/graphvis/requirements.txt`, `repro/graphvis/gv/cli.py` (lệnh `download`)

**Interfaces:**
- Consumes: `repro.common.remote_zip.extract_members`, `repro.common.paths.data_root`
- Produces: `download_qagnn(dest: Path) -> dict[str, Path]` trả về khóa `concept_txt`, `pruned_graph`, `csqa_grounded_{train,dev}`, `obqa_grounded_{train,test}`, `csqa_{train,dev}_jsonl`; `load_qa(dataset: Literal["csqa","obqa"], split) -> list[QAItem]` với `QAItem(id: str, question: str, choices: list[str], labels: list[str], answer: str)`.

- [ ] Step 1: Viết `download_qagnn` chỉ trích các member: `cpnet/concept.txt`, `cpnet/conceptnet.en.pruned.graph`, `csqa/grounded/{train,dev}.grounded.jsonl`, `obqa/grounded/{train,test}.grounded.jsonl`, `csqa/{train,dev}_rand_split.jsonl` (≈60 MB nén) từ `https://nlp.stanford.edu/projects/myasu/QAGNN/data_preprocessed_release.zip`; bỏ qua nếu file đã có.
- [ ] Step 2: `load_qa`: CSQA từ HF `tau/commonsense_qa` (train/validation), OBQA từ HF `allenai/openbookqa` config `main` (train/test). Test `test_load_qa_shapes` dùng file jsonl giả 2 dòng (không mạng) cho hàm parse `_parse_hf_row`.
- [ ] Step 3: `requirements.txt`: `transformers>=4.45,<5`, `peft>=0.13`, `bitsandbytes>=0.44`, `accelerate>=1.0`, `datasets>=2.20`, `graphviz`, `networkx>=3`, `sentence-transformers>=3`, `wandb`, `pyyaml`, `pillow`, `pytest`. (Ghi phiên bản cố định sau lần chạy Colab đầu tiên.)
- [ ] Step 4: Commit.

### Task 2: Truy hồi đồ thị con (G0)

**Files:** Create `gv/data/retrieve.py`, `tests/test_retrieve.py`

**Interfaces:**
- Produces:
  - `MERGED_RELATIONS: list[str]` (17 quan hệ, đúng thứ tự của `GraphVis-code/generate_data.py:15-33`).
  - `load_cpnet(concept_txt: Path, graph_path: Path) -> CPNet` (dataclass: `id2concept: list[str]`, `concept2id: dict`, `graph: nx.MultiDiGraph`).
  - `build_question_graph(cp: CPNet, grounded_rows: list[dict], max_nodes: int = 80) -> SubGraph` với `SubGraph(nodes: list[str], triples: list[tuple[str,str,str]], qc: set[str], ac: set[str])`; tên nút thay `_` bằng khoảng trắng.
  - Quy tắc quan hệ: lấy rel id ≤16 duy nhất giữa u→v; nếu >1 và có 15 (`related_to`) thì bỏ 15; lấy cái đầu (giống `get_edge`). Hướng cạnh theo ConceptNet (u→v có cạnh mới thêm).

- [ ] **Step 1: Test (đồ thị đồ chơi, không cần mạng)**

```python
import networkx as nx
from gv.data.retrieve import CPNet, build_question_graph, MERGED_RELATIONS

def toy():
    names = ["revolving_door", "security", "bank", "door", "library", "building"]
    g = nx.MultiDiGraph()
    rid = {r: i for i, r in enumerate(MERGED_RELATIONS)}
    for u, r, v in [("revolving_door","is_a","door"), ("door","at_location","bank"),
                    ("security","at_location","bank"), ("door","part_of","building"),
                    ("library","is_a","building")]:
        g.add_edge(names.index(u), names.index(v), rel=rid[r], weight=1.0)
    return CPNet(id2concept=names, concept2id={n:i for i,n in enumerate(names)}, graph=g)

def test_two_hop_path_adds_intermediate():
    sg = build_question_graph(toy(), [{"qc":["revolving_door","security"], "ac":["bank"]}])
    assert "door" in sg.nodes                       # revolving_door -> door -> bank
    assert ("door", "at location", "bank") in sg.triples
    assert ("security", "at location", "bank") in sg.triples

def test_merges_all_choices():
    sg = build_question_graph(toy(), [{"qc":["revolving_door"], "ac":["bank"]},
                                      {"qc":["revolving_door"], "ac":["library"]}])
    assert {"bank", "library"} <= set(sg.nodes)

def test_empty_grounding_yields_placeholder_graph():
    sg = build_question_graph(toy(), [{"qc": [], "ac": []}])
    assert sg.nodes == [] and sg.triples == []

def test_max_nodes_keeps_question_and_answer_concepts():
    sg = build_question_graph(toy(), [{"qc":["revolving_door","security"], "ac":["bank"]}], max_nodes=3)
    assert {"revolving door", "security", "bank"} <= set(sg.nodes) and len(sg.nodes) == 3
```

- [ ] Step 2: chạy → FAIL. Step 3: cài đặt (2-hop: với mỗi q∈qc, a∈ac trong đồ thị vô hướng `cpnet_simple`: thêm a nếu kề q; thêm x ∈ N(q)∩N(a); đếm số đường đi qua mỗi x; khi vượt `max_nodes` giữ qc∪ac rồi các x có số đường đi giảm dần, hòa thì theo thứ tự chữ cái). Step 4: PASS. Step 5: commit.

### Task 3: Vẽ ảnh + 6 tác vụ D_g + D_v + text-only (G0)

**Files:** Create `gv/data/render.py`, `gv/data/tasks.py`, `gv/data/build.py`, `tests/test_tasks.py`, `tests/test_render.py`

**Interfaces:**
- Consumes: `SubGraph`, `QAItem`.
- Produces:
  - `render_graph(sg: SubGraph, out_png: Path, size_px: int | None = None, dpi: int = 96) -> Path` (graphviz `Digraph`, layout `dot`, nút ellipse, nhãn cạnh = quan hệ; đồ thị rỗng → ảnh một nút "no relevant knowledge"; `size_px` ≠ None → thu nhỏ cạnh dài về `size_px` bằng PIL, dùng cho G7/C2).
  - `TASKS = ("node_description","node_degree","highest_degree","node_count","edge_count","triple_listing")`; `OCR_TASKS = ("node_description","triple_listing")`; `GRAPH_TASKS` = 4 tác vụ còn lại.
  - `PROMPTS: dict[str, list[str]]` — 5 mẫu câu/tác vụ, lấy nguyên văn Phụ lục A của bài (đã sửa 2 lỗi `generate_data.py:54,60`).
  - `make_task_sample(task: str, sg: SubGraph, rng: random.Random) -> tuple[str, str]` (câu hỏi, đáp án): bậc tính trên **cùng tập cạnh** với số cạnh (= `len(sg.triples)`, mỗi bộ ba đóng góp 1 bậc cho mỗi đầu) — sửa điểm không nhất quán §A3.
  - `QA_PROMPT = "The image represents a knowledge graph relevant to the question, which may or may not be useful. Question: {question}\n{options}\nAnswer with the option's letter from the given choices directly."`; `format_options(item) -> "A. ...\nB. ..."`.
  - `build_dataset(kind: Literal["dg","dv","text"], items, graphs, img_dir, out_jsonl, tasks=TASKS, seed=42, max_items=None) -> int`; mỗi dòng jsonl: `{"id","image"(path|None),"prompt","answer","task","n_nodes","n_edges"}`.

- [ ] **Step 1: Test**

```python
import random
from gv.data.retrieve import SubGraph
from gv.data.tasks import make_task_sample, PROMPTS, TASKS

SG = SubGraph(nodes=["choker","jewelry","box","store"],
              triples=[("choker","is a","jewelry"),("jewelry","at location","box"),("box","used for","store")],
              qc={"choker"}, ac={"store"})

def test_prompt_pools_have_five_distinct_templates():
    for t in TASKS:
        assert len(set(PROMPTS[t])) == 5

def test_counts_and_degrees_are_consistent():
    q, a = make_task_sample("edge_count", SG, random.Random(0)); assert a.endswith("3 edges in the graph.")
    q, a = make_task_sample("node_count", SG, random.Random(0)); assert "4 nodes" in a
    q, a = make_task_sample("highest_degree", SG, random.Random(0))
    assert ('"jewelry"' in a or '"box"' in a) and "degree of 2" in a

def test_node_degree_prompt_substitutes_name():
    for seed in range(20):
        q, a = make_task_sample("node_degree", SG, random.Random(seed))
        assert "{node" not in q

def test_triple_listing_contains_all_triples():
    q, a = make_task_sample("triple_listing", SG, random.Random(0))
    for h, r, t in SG.triples:
        assert f"({h}, {r}, {t})" in a
```

`test_render.py`: `@pytest.mark.skipif(shutil.which("dot") is None, ...)` vẽ SG ra PNG, kiểm tra file tồn tại và `size_px=50` cho ảnh cạnh dài 50.
- [ ] Step 2 FAIL → Step 3 cài → Step 4 PASS → Step 5 commit.
- [ ] Step 6: lệnh CLI `python -m gv.cli build-data --dataset csqa --split train --scale {full|lite} [--image_size 50]` → tạo `graphs.jsonl` (SubGraph + thống kê), `images/`, `dg.jsonl`, `dv.jsonl`, `text.jsonl`; log W&B run `G0-data-csqa`: histogram số nút/cạnh (so với Fig 4–5), số đồ thị rỗng, 20 ảnh mẫu (`wandb.Image`).

### Task 4: KAPING

**Files:** Create `gv/data/kaping.py`, `tests/test_kaping.py`
**Interfaces:** `top_k_triples(question: str, triples: list[tuple], k: int = 10, encoder=None) -> list[tuple]` (encoder mặc định `sentence-transformers/all-mpnet-base-v2`; test truyền encoder giả); `kaping_prefix(triples) -> "Below are facts in the form of the triple meaningful to answer the question.\n(h, r, t)\n..."`.
- [ ] Test với encoder giả (vector one-hot theo từ khóa) → triple chứa từ trong câu hỏi xếp đầu; `k` lớn hơn số triple → trả hết. FAIL → cài → PASS → commit.

### Task 5: Nạp mô hình + collate (mask loss)

**Files:** Create `gv/model/load.py`, `gv/model/collate.py`, `tests/test_collate.py`
**Interfaces:**
- `load_model_and_processor(model_id="llava-hf/llava-v1.6-mistral-7b-hf", image_mode="pad", quant: Literal["none","4bit"]="none", lora: dict | None = None, adapter_path: str | None = None, dtype="bf16") -> (model, processor)`:
  - `image_mode="pad"`: `processor.image_processor.image_grid_pinpoints = [[336,336]]`, `model.config.image_grid_pinpoints = [[336,336]]`; `pad_to_square(img)` áp dụng trong collate/generate.
  - `lora`: `LoraConfig(r=128, lora_alpha=256, lora_dropout=0.05, target_modules=<mọi nn.Linear trong language_model>, modules_to_save=["multi_modal_projector"])`; vision tower `requires_grad=False`.
  - `quant="4bit"`: `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=bf16|fp16)` + `prepare_model_for_kbit_training`.
- `build_messages(prompt: str, has_image: bool) -> list[dict]`; `Collator(processor, max_len=2048, image_mode="pad")`: tạo `input_ids`, `pixel_values`, `image_sizes`, `labels` với `labels = -100` cho mọi token trước câu trả lời (tìm vị trí bằng cách tokenize riêng phần prompt-đã-áp-template rồi lấy độ dài). Hàm thuần `mask_prompt_tokens(input_ids: list[int], prompt_len: int, pad_id: int) -> list[int]` để test được không cần transformers.
- [ ] Test `mask_prompt_tokens`: prompt_len=5, chuỗi 9 token, 1 pad cuối → labels = [-100]*5 + 3 token trả lời + [-100]. FAIL → cài → PASS → commit.
- [ ] Test Colab (đánh dấu `@pytest.mark.gpu`, chỉ chạy trên Colab): nạp model 4bit, một batch 2 mẫu, kiểm tra `loss` hữu hạn và số tham số trainable ≈ 336M + 21M (sai số 5%).

### Task 6: Huấn luyện theo giai đoạn

**Files:** Create `gv/train.py`, `configs/*.yaml`
**Interfaces:**
- Config YAML:
```yaml
exp_id: G1-csqa-curriculum
dataset: csqa
image_mode: pad
quant: none            # 4bit cho L4/T4
lr: 2.0e-5
projector_lr: 2.0e-5
global_batch: 4
per_device_batch: 2
epochs_per_stage: 1
max_len: 2048
seed: 42
stages:                # chạy tuần tự, giai đoạn sau khởi tạo từ adapter giai đoạn trước
  - {name: stage1, data: [dg]}
  - {name: stage2, data: [dv]}
```
  Các biến thể: `G4_joint` → `stages: [{name: joint, data: [dg, dv], shuffle: true}]`; `G4_graph_first` → `[{data:[dg_graph]}, {data:[dg_ocr]}, {data:[dv]}]`; `G4_ocr_first` ngược lại; `G7_lowres` = curriculum với `image_size: 50`; `C3_text_only` → `[{data:[text]}]` (không ảnh); `G1_obqa_curriculum` dataset obqa. (`dg_graph`/`dg_ocr` = lọc `task` theo `GRAPH_TASKS`/`OCR_TASKS`.)
- `run_stages(cfg: dict) -> Path` (thư mục adapter cuối). HF `Trainer` với `report_to="wandb"`, `save_steps=500`, `save_total_limit=2`, `lr_scheduler_type="cosine"`, `warmup_ratio=0.03`, `weight_decay=0`, optimizer AdamW, nhóm tham số riêng cho projector (`projector_lr`); resume: nếu `out/<stage>/checkpoint-*` có thì `trainer.train(resume_from_checkpoint=True)`; nếu `out/<stage>/final` có thì bỏ qua stage.
- [ ] Test CPU: `plan_stages(cfg) -> list[list[str]]` cho 5 biến thể (thuần Python) — FAIL → cài → PASS.
- [ ] Lệnh: `python -m gv.cli train --config configs/G1_csqa_curriculum.yaml [--dry_run]` (`--dry_run`: 16 mẫu/stage, 2 bước). Commit.

### Task 7: Đánh giá CSQA/OBQA + Table 3 + Table 4

**Files:** Create `gv/eval/generate.py`, `gv/eval/mcqa.py`, `gv/eval/graph_reading.py`, `tests/test_metrics.py`
**Interfaces:**
- `extract_choice_letter(text: str, n_choices: int) -> str | None`.
- `eval_mcqa(model, processor, dataset, split, mode: Literal["text","image","kaping","image+kaping"], adapter=None, image_size=None, exp_id=...) -> dict` → log W&B: `accuracy`, bảng dự đoán (id, gold, pred, raw), và `accuracy_by_nodes` (bins ≤10, 11–20, 21–40, >40) — dùng lại cho C1.
- `score_graph_task(task, pred: str, sg: SubGraph, gold: str) -> float`: số/tên → exact match sau chuẩn hóa (trích số nguyên đầu tiên; nút bậc cao nhất: đúng nếu tên thuộc tập nút có bậc max **và** bậc đúng); `node_description`/`triple_listing` → tỉ lệ phần tử gold xuất hiện trong pred (trung bình theo mẫu, đúng mô tả "average accuracy of each test example").
- [ ] **Test**

```python
from gv.eval.mcqa import extract_choice_letter
from gv.eval.graph_reading import score_graph_task

def test_extract_choice_letter_variants():
    assert extract_choice_letter("B", 5) == "B"
    assert extract_choice_letter("The answer is (C).", 5) == "C"
    assert extract_choice_letter("b. bank", 5) == "B"
    assert extract_choice_letter("E", 4) is None
    assert extract_choice_letter("", 5) is None

def test_graph_scores():
    assert score_graph_task("node_count", "There are 4 nodes.", SG, "There are 4 nodes in the graph.") == 1.0
    assert score_graph_task("node_count", "There are 5 nodes.", SG, "There are 4 nodes in the graph.") == 0.0
    s = score_graph_task("triple_listing", "(choker, is a, jewelry)", SG, "")
    assert abs(s - 1/3) < 1e-9
```
(`SG` như Task 3.) FAIL → cài → PASS → commit.
- [ ] CLI: `eval-mcqa --dataset csqa --split dev --mode image --adapter <path>`; `eval-graph-reading --adapter <stage1_path|none>` (tập kiểm tra = D_g dựng từ CSQA dev, 1 mẫu/tác vụ/ảnh).

### Task 8: VQA + độ phức tạp ảnh

**Files:** Create `gv/eval/vqa.py`, `gv/eval/complexity.py`
**Interfaces:**
- `mmbench_circular_accuracy(rows: list[dict]) -> float`: nhóm theo `index % 1_000_000`; nhóm đúng khi mọi dòng trong nhóm đúng.
- `pope_metrics(preds: list[str], golds: list[str]) -> dict(acc, f1, precision, recall, yes_ratio)` (pred "yes" nếu bắt đầu bằng yes sau lower/strip).
- `eval_scienceqa_img`, `eval_mmbench`, `eval_pope` (HF `derek-thomas/ScienceQA` test lọc có ảnh; `lmms-lab-encoder/MMBench` config `en` split `dev`; `lmms-lab-encoder/POPE` config `Full` 3 split). Prompt kiểu LLaVA: ScienceQA/MMBench "…\nAnswer with the option's letter from the given choices directly."; POPE "…\nAnswer the question using a single word or phrase."
- `run_resolution_sweep(adapter, sizes=(50,112,224,336,672), image_modes=("pad","anyres"))` → bảng accuracy CSQA dev × kích thước (render lại ảnh với `size_px`).
- [ ] Test `mmbench_circular_accuracy` (2 nhóm, 1 nhóm sai 1 bản xoay → 0.5) và `pope_metrics` (ví dụ tay). FAIL → cài → PASS → commit.

### Task 9: Notebook GraphVis + README + DEVIATIONS

**Files:** Create `repro/graphvis/notebooks/GraphVis_Colab.ipynb`, `repro/graphvis/README.md`, `repro/graphvis/DEVIATIONS.md`; Modify `docs/experiments.md`, `README.md` (gốc)

Notebook: ô 0 chọn GPU & `SCALE`, `QUANT`; ô 1 mount Drive, clone repo, `apt-get install graphviz`, `pip install -r`; ô 2 `wandb login`; rồi **một ô cho mỗi ID** theo đúng thứ tự phụ thuộc:

| Ô | Lệnh | Phụ thuộc |
|---|---|---|
| G0 | `build-data` csqa train/dev, obqa train/test | – |
| G1a | `eval-mcqa` base: text, image, kaping (csqa dev, obqa test) | G0 |
| G1b | `train G1_csqa_curriculum` rồi `eval-mcqa image` | G0 |
| G1c | `train G1_obqa_curriculum` rồi `eval-mcqa image` (obqa) | G0 |
| G6 | `eval-graph-reading` base và adapter stage1 của G1b | G1b |
| G2 | `eval-vqa all` base và G1b | G1b |
| G4 | `train` joint / graph_first / ocr_first + `eval-mcqa image` | G0 |
| G3 | `eval-vqa scienceqa` cho joint | G4 joint |
| G5 | `eval-mcqa image+kaping` cho base(=kaping), joint, curriculum | G1b, G4 |
| G7 | `build-data --image_size 50` + `train G7_lowres` + eval | G0 |
| C3 | `train C3_text_only` + `eval-mcqa text` | G0 |
| C1 | đọc lại bảng dự đoán của G1a/G1b/C3 → `accuracy_by_nodes` (+ graph reading theo bin) | G1b, C3, G6 |
| C2 | `resolution-sweep` cho base và G1b | G1b |

- [ ] Kiểm tra JSON notebook hợp lệ: `python -c "import json;json.load(open('repro/graphvis/notebooks/GraphVis_Colab.ipynb',encoding='utf-8'))"`.
- [ ] Commit.

---

## Phần B — SciMKG

### Task 10: Dữ liệu SciMKG + chuẩn hóa

**Files:** Create `smkg/data.py`, `smkg/normalize.py`, `tests/test_normalize.py`, `tests/test_data.py`
**Interfaces:**
- `download_subset(dest: Path, lessons: list[str] | None = None) -> Path`: tải `Groundtruth.zip` (0,1 MB) và trích từ `RawData.zip` (record Zenodo 19691725) qua HTTP Range **chỉ** các bài có nhãn vàng: `<bài>.txt` (đoạn có mốc thời gian), `*.png`, `<bài>_Extractionner|_Validationner|_Intergrationner` (kết quả gốc của tác giả). Lưu `manifest.json` (bài → môn, đường dẫn).
- `Lesson(name, subject: Literal["biology","physics","chemistry"], segments: list[tuple[str, str]], images: list[Path], gt: set[str], author_final: set[str] | None)`.
- `parse_segments(text: str) -> list[tuple[str,str]]` cho dòng `"mm:ss-mm:ss": "nội dung"`.
- `normalize_concept(s: str) -> str`: NFKC, bỏ khoảng trắng, bỏ dấu câu đầu/cuối, bỏ tiền tố `-`/`•`.
- [ ] **Test**

```python
from smkg.normalize import normalize_concept
from smkg.data import parse_segments

def test_normalize_fullwidth_and_spaces():
    assert normalize_concept(" 细胞膜 ") == normalize_concept("细胞膜")
    assert normalize_concept("ＤＮＡ") == "DNA"
    assert normalize_concept("- 叶绿体。") == "叶绿体"

def test_parse_segments():
    txt = '"00:11-01:18": "各位同学大家好"\n\n"01:18-03:13": "1959年，科学家"\n'
    assert parse_segments(txt) == [("00:11-01:18", "各位同学大家好"), ("01:18-03:13", "1959年，科学家")]
```
FAIL → cài → PASS → commit. CLI: `python -m smkg.cli download [--lessons N]` + log W&B `S0-data` (số bài/môn, số đoạn, số ảnh, kích thước nhãn vàng).

### Task 11: LLM client + prompt

**Files:** Create `smkg/llm.py`, `smkg/prompts.py`, `configs/models.yaml`, `tests/test_llm.py`
**Interfaces:**
- `models.yaml`:
```yaml
gpt-4o:            {provider: openai,   model: gpt-4o-2024-11-20, base_url: null,   key_env: OPENAI_API_KEY}
deepseek-v3:       {provider: deepseek, model: deepseek-chat,     base_url: https://api.deepseek.com/v1, key_env: DEEPSEEK_API_KEY}
gemini-1.5-flash:  {provider: gemini,   model: gemini-1.5-flash,  base_url: https://generativelanguage.googleapis.com/v1beta/openai/, key_env: GEMINI_API_KEY, fallback: gemini-2.0-flash}
qwen3-14b:         {provider: local, model: qwen3:14b,  base_url: http://localhost:11434/v1, key_env: null}
# ... qwen3-4b/8b, gemma3-4b/12b/27b, phi3-mini/small/medium
```
- `LLMClient(name: str, cfg: dict, cache_dir: Path)`; `.chat(prompt: str, **params) -> LLMReply(text: str | None, error: Literal[None,"no_response","timeout","api_error"], cached: bool)`; cache = file jsonl/sqlite khóa `sha256(model|prompt|params)`; retry 3 lần (backoff); **không bao giờ raise** với lỗi mạng/API. `check_models(names) -> dict[name, status|fallback_used]`.
- `parse_concepts(text: str | None) -> tuple[list[str], Literal["ok","format_error","no_response"]]`: lấy các dòng `^\s*[-•]\s*(.+)$`; không có dòng nào nhưng text khác rỗng → `format_error`.
- `prompts.py`: `EXTRACT`, `FEEDBACK`, `REFINE`, `IMAGE_ALIGN` chép **nguyên văn từ mã tác giả** (`SciMKG/ExtractMultimodalConcepts/NER.py:70-80,170-185,213-242`, `ImageAligment.py:16-52`) với chỗ điền `{subject}`, `{candidates}`, `{feedback}`.
- [ ] **Test**

```python
from smkg.llm import parse_concepts, LLMClient

def test_parse_concepts_handles_garbage():
    assert parse_concepts("-细胞膜\n- 磷脂\n") == (["细胞膜", "磷脂"], "ok")
    assert parse_concepts("Here are the concepts: none")[1] == "format_error"
    assert parse_concepts(None)[1] == "no_response"
    assert parse_concepts("")[1] == "no_response"

def test_client_counts_no_response(tmp_path, monkeypatch):
    c = LLMClient("fake", {"provider": "fake"}, tmp_path)
    monkeypatch.setattr(c, "_call", lambda prompt, **p: (_ for _ in ()).throw(TimeoutError()))
    r = c.chat("x")
    assert r.text is None and r.error in ("timeout", "api_error")
    assert c.counters["no_response"] + c.counters["timeout"] + c.counters["api_error"] == 1

def test_cache_hit(tmp_path, monkeypatch):
    c = LLMClient("fake", {"provider": "fake"}, tmp_path)
    calls = []
    monkeypatch.setattr(c, "_call", lambda prompt, **p: calls.append(1) or "-a")
    c.chat("p"); r = c.chat("p")
    assert r.cached and len(calls) == 1
```
FAIL → cài → PASS → commit. CLI: `python -m smkg.cli check-models` (in bảng model khả dụng/thay thế → dán vào DEVIATIONS.md).

### Task 12: EVIA với log phiếu + tích hợp offline

**Files:** Create `smkg/evia.py`, `tests/test_evia.py`
**Interfaces:**
- `run_extraction(lesson, clients: dict[str, LLMClient], extractors: list[str], seen: set[str]) -> dict[seg_id, dict[model, list[str]]]` (tập `seen` toàn cục theo bài → khái niệm gắn đoạn đầu tiên, đúng hành vi mã gốc).
- `run_validation(candidates: dict[seg_id, list[str]], clients, validators: list[str], K: int = 5) -> VoteLog` với `VoteLog = dict[seg_id, dict[model, list[list[str]]]]` (danh sách sống sót của từng vòng 1..K). Vòng k: FEEDBACK(danh sách gốc, phản hồi tích lũy) → REFINE(danh sách gốc, phản hồi vòng k).
- `integrate(candidates, votes: VoteLog, models: list[str], K: int, alpha: float) -> dict[seg_id, dict[concept, float]]` → điểm = (số vòng sống sót trên `models` × vòng 1..K) / (len(models)·K); giữ nếu ≥ α. Trả cả điểm để tính AUC.
- Ablation (không gọi thêm API): `full` = extractors 3 LLM (hợp) + validators 3 LLM; `wo_MLs` = chỉ GPT-4o ở cả E và V (dùng lại log); `wo_SFFD` = 3 LLM chỉ trích xuất, điểm = tỉ lệ LLM trích ra khái niệm, ngưỡng α; `wo_both` = GPT-4o trích xuất trực tiếp. Table 5 = `integrate` với K∈{1,3,5}, α∈{0.5,0.7,0.9} trên log `full`.
- [ ] **Test**

```python
from smkg.evia import integrate

def test_integrate_normalizes_votes():
    cands = {"s1": ["上皮组织", "系统"]}
    votes = {"s1": {"m1": [["上皮组织"], ["上皮组织"]], "m2": [["上皮组织", "系统"], ["上皮组织"]]}}
    out = integrate(cands, votes, ["m1", "m2"], K=2, alpha=0.7)
    assert out["s1"]["上皮组织"] == 1.0
    assert abs(out["s1"]["系统"] - 0.25) < 1e-9

def test_integrate_prefix_rounds_for_table5():
    cands = {"s1": ["a"]}
    votes = {"s1": {"m": [["a"], [], [], [], []]}}
    assert integrate(cands, votes, ["m"], K=1, alpha=0.5)["s1"]["a"] == 1.0
    assert integrate(cands, votes, ["m"], K=5, alpha=0.5)["s1"]["a"] == 0.2
```
FAIL → cài → PASS → commit. CLI: `python -m smkg.cli evia --models gpt-4o,deepseek-v3,gemini-1.5-flash --K 5 [--lessons N] [--dry_run]` → lưu `votes.json`, `extraction.json`, counters.

### Task 13: Metric + phân loại lỗi + kết quả gốc của tác giả

**Files:** Create `smkg/metrics.py`, `smkg/errors.py`, `tests/test_metrics.py`
**Interfaces:**
- `set_prf(pred: dict[lesson,set], gold: dict[lesson,set]) -> dict(precision, recall, f1)` (micro).
- `universe_metrics(pred, gold, pool, scores=None) -> dict(acc, mcc, auc)` (D9; MCC = 0 khi mẫu số 0; AUC bằng `sklearn.metrics.roc_auc_score` nếu có điểm, ngược lại balanced accuracy).
- `classify_errors(lesson_pred, gold, reply_status) -> Counter` với FE (format_error), NR (no_response/timeout/api_error), IO (khái niệm dự đoán không xuất hiện trong văn bản đoạn sau chuẩn hóa — ảo giác/sai ngôn ngữ), CE (dự đoán có trong văn bản nhưng không thuộc nhãn vàng), CO (nhãn vàng bị bỏ sót). Định nghĩa ghi vào DEVIATIONS.md.
- [ ] **Test**

```python
from smkg.metrics import set_prf, universe_metrics

def test_set_prf_micro():
    m = set_prf({"l1": {"a", "b"}, "l2": {"c"}}, {"l1": {"a"}, "l2": {"c", "d"}})
    assert m["precision"] == 2/3 and m["recall"] == 2/3

def test_universe_metrics_perfect():
    m = universe_metrics({"l": {"a"}}, {"l": {"a"}}, {"l": {"a", "x"}})
    assert m["acc"] == 1.0 and m["mcc"] == 1.0
```
FAIL → cài → PASS → commit. CLI `score --run <dir>` và `score-authors` (chấm file `_Intergrationner` gốc của tác giả với nhãn vàng → mốc tham chiếu `S1-authors-output`, không tốn API).

### Task 14: Baseline + mô hình mở theo kích thước

**Files:** Create `smkg/baselines.py`, `tests/test_baselines.py`
**Interfaces:** mỗi baseline `run_<name>(lesson, client, subject) -> dict[seg_id, list[str]]`:
- `decomposed_qa`: hỏi từng đoạn "Những khái niệm {subject} nào xuất hiện…" kèm gợi ý cú pháp (danh từ/cụm danh từ đã tách bởi LLM ở lượt trước) — theo Xie et al. 2023 (hai lượt: tách cụm danh từ → hỏi loại).
- `gpt_ner`: yêu cầu viết lại đoạn, bao khái niệm bằng `@@…##`, trích bằng regex `@@(.+?)##`; kiểm chứng tự thân (self-verification) 1 lượt.
- `self_improving`: tự gán nhãn mọi đoạn (5 lần lấy mẫu, nhiệt độ 0.7, bỏ phiếu đa số) → chọn 4 ví dụ tin cậy nhất làm few-shot theo độ tương tự (sentence-transformers đa ngôn ngữ) → suy luận.
- `universal_ner`: model `Universal-NER/UniNER-7B-type` (HF, chạy local GPU, prompt chuẩn UniNER với nhãn "{subject} concept").
- LinkNER: không cài (D11).
- [ ] Test: `extract_gptner_spans("…@@细胞膜##由@@磷脂##…") == ["细胞膜","磷脂"]`. FAIL → cài → PASS → commit.
- [ ] Mô hình mở cho bảng kích thước: notebook cài Ollama trên Colab, `ollama pull qwen3:{4b,8b,14b}`, `gemma3:{4b,12b,27b}`, `phi3:{mini,medium}`; Phi-3-small (7B) không có trên Ollama → chạy bằng vLLM OpenAI server (`vllm serve microsoft/Phi-3-small-8k-instruct --trust-remote-code`) và trỏ `base_url` tương ứng. Bộ ba = (Qwen3-4B, Gemma3-4B, Phi-3-mini), (8B, 12B, small), (14B, 27B, medium).

### Task 15: Đánh giá căn chỉnh ảnh (Table 7) + phiếu chấm người

**Files:** Create `smkg/align_eval.py`, `tests/test_align.py`
**Interfaces:**
- `load_pairs(scimkg_zip_dir, raw_dir, lessons) -> list[Pair(concept_zh, concept_en, image_path, lesson)]` (lấy từ `SciMKG.zip` trên Zenodo — trường `png` của khái niệm; tên tiếng Anh: dịch bằng LLM có cache, vì CLIP/BLIP chỉ hiểu tiếng Anh — ghi DEVIATIONS).
- `clip_score(pairs, model="openai/clip-vit-large-patch14") -> dict(raw_cos_mean, clipscore_w2.5_mean)`; `blip_itm(pairs, model="Salesforce/blip-itm-large-coco") -> mean_prob`; `vqascore(pairs, model="clip-flant5-xl")` (tùy chọn, `t2v_metrics`).
- `regenerate_alignment(lessons, client="gpt-4o", runs=3)` (tùy chọn, tái tạo căn chỉnh bằng prompt C1–C4) → so sánh tập khái niệm–ảnh với bản gốc (P/R).
- `make_human_sheet(pairs, n=100, seed=42, out_csv)` → CSV cột `pair_id, concept_zh, concept_en, image_file, rater_khang, rater_nhi`; `score_human_sheet(csv) -> mean, cohen_kappa`.
- [ ] Test `make_human_sheet` lấy đúng n cặp, ổn định theo seed; `score_human_sheet` trên CSV tay. FAIL → cài → PASS → commit.

### Task 16: Notebook SciMKG + README + DEVIATIONS + cập nhật docs

**Files:** Create `repro/scimkg/notebooks/SciMKG_Colab.ipynb`, `repro/scimkg/README.md`, `repro/scimkg/DEVIATIONS.md`; Modify `docs/experiments.md` (cột lệnh + phụ thuộc), `README.md` gốc (mục tái lập), `CLAUDE.md` §5.

| Ô | Lệnh | GPU/API |
|---|---|---|
| S0 | `download` + `check-models` + `score-authors` | không |
| S1 | `evia --models gpt-4o,deepseek-v3,gemini-1.5-flash --K 5` → `score` | API |
| S1b | baselines: decomposed_qa, gpt_ner, self_improving (GPT-4o), universal_ner (GPU) | API + GPU |
| S2 | `ablate wo_MLs wo_SFFD wo_both` (dùng lại log S1, chỉ wo_both/wo_SFFD gọi thêm E) | API ít |
| S3 | `evia` với 3 bộ mô hình mở | GPU |
| S4 | `grid --K 1,3,5 --alpha 0.5,0.7,0.9` trên log S1 | không |
| S5 | `errors` (từ log S1) + `align-eval clip,blip[,vqascore]` + `human-sheet` | GPU nhẹ |

- [ ] Kiểm tra JSON notebook; chạy toàn bộ test local `cd repro && python -m pytest -q`; commit; push nhánh `khang/repro-scaffold`, mở PR (người dùng tự merge).

---

## Self-review (đã chạy)

- Phủ spec: G0 (T1–3), G1 (T6–7, notebook), G2 (T8), G3 (T8+notebook), G4 (T6 configs), G5 (T4, T7), G6 (T7), G7 (T3 `size_px`, T6 config), C1 (T7 `accuracy_by_nodes`), C2 (T8), C3 (T3 `text`, T6 config); S0 (T10), S1 (T12–14), S2 (T12), S3 (T14 + T12), S4 (T12 `integrate`), S5 (T13 errors, T15). README/DEVIATIONS (T9, T16). AGENTS.md (T0).
- Giới hạn đã biết và đã ghi (không phải lỗ hổng kế hoạch): LinkNER bỏ (D11), X-VLM/VQAScore tùy chọn (D12), số liệu MMBench/POPE chấm local (D6).
- Tên hàm thống nhất giữa các task: `build_question_graph`, `SubGraph`, `make_task_sample`, `load_model_and_processor`, `run_stages`, `eval_mcqa`, `score_graph_task`, `LLMClient.chat`, `parse_concepts`, `integrate`, `set_prf`, `universe_metrics`.
