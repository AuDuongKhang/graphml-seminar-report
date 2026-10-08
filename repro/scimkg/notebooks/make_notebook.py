"""Sinh SciMKG_Repro.ipynb. Sửa file này rồi chạy lại: python repro/scimkg/notebooks/make_notebook.py"""
import json
from pathlib import Path

CELLS = []


def md(s):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md("""
# Tái lập SciMKG — Kaggle / Colab

Không có API key: mặc định dùng bộ ba mô hình mở lớn nhất của Bảng 4 trong bài (Qwen3-14B + Gemma3-27B +
Phi-3-medium) qua **Ollama** trên GPU miễn phí (Kaggle T4 ×2 khuyến nghị: 32 GB cho Gemma3-27B q4).
Có key thì đặt `USE_PAPER_MODELS = True` (GPT-4o, DeepSeek-V3, Gemini-1.5-Flash — xem `check-models`).

| Ô | Thực nghiệm | Mục báo cáo |
|---|---|---|
| S0 | tải dữ liệu, kiểm tra mô hình, chấm lại đầu ra gốc của tác giả | XII-A, XII-B |
| S1 | EVIA đầy đủ | XII-B |
| S1b | baseline (Decomposed-QA, GPT-NER, Self-Improving, UniversalNER) | XII-B |
| S2 | ablation w/o MLs, w/o SF-FD, w/o cả hai | XII-B |
| S3 | bộ mô hình mở theo kích thước | XII-B |
| S4 | lưới K × α | XII-C |
| S5 | phân loại lỗi, CLIP/BLIP, chấm người, căn chỉnh lại ảnh | XII-D |

Secrets (Kaggle *Add-ons → Secrets* / Colab 🔑): `WANDB_API_KEY`; tùy chọn `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `GEMINI_API_KEY`.
Kaggle: bật *Persistence → Files only* để giữ cache lời gọi LLM và log từng bài giữa các phiên.
""")

code("""
# ==== CẤU HÌNH ====
LESSONS = 30                 # số bài đánh giá (chia đều 3 môn); 0 = cả 98 bài
USE_PAPER_MODELS = False     # True nếu có đủ 3 API key
PAPER = ["gpt-4o", "deepseek-v3", "gemini-1.5-flash"]
OPEN = {"small": ["qwen3-4b", "gemma3-4b", "phi3-mini"],
        "mid":   ["qwen3-8b", "gemma3-12b", "phi3-mini"],    # bài dùng Phi-3-small (7B): xem ô vLLM tùy chọn
        "large": ["qwen3-14b", "gemma3-27b", "phi3-medium"]}
MAIN = PAPER if USE_PAPER_MODELS else OPEN["large"]
MAIN_RUN = "S1-paper-models" if USE_PAPER_MODELS else "S1-open-large"
WORKERS = 4
WANDB_PROJECT = "graphml-seminar-report"
REPO_URL = "https://github.com/AuDuongKhang/graphml-seminar-report.git"
BRANCH = "main"
""")

code("""
import os, sys, subprocess, time
ON_KAGGLE = os.path.exists("/kaggle/working")
ON_COLAB = "google.colab" in sys.modules
if ON_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE = "/content/drive/MyDrive/graphml-repro"
elif ON_KAGGLE:
    BASE = "/kaggle/working/graphml-repro"
else:
    BASE = os.path.abspath("graphml-repro")
os.makedirs(BASE, exist_ok=True)
os.environ.update(GRAPHML_DATA=f"{BASE}/data", GRAPHML_OUT=f"{BASE}/out", WANDB_PROJECT=WANDB_PROJECT)
REPO = f"{BASE}/repo"
if not os.path.exists(REPO):
    subprocess.run(["git", "clone", "-b", BRANCH, REPO_URL, REPO], check=True)
else:
    subprocess.run(["git", "-C", REPO, "pull"], check=True)
SM = f"{REPO}/repro/scimkg"
!pip install -q -r {SM}/requirements.txt

def secret(name):
    try:
        if ON_KAGGLE:
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret(name)
        if ON_COLAB:
            from google.colab import userdata
            return userdata.get(name)
    except Exception:
        return os.environ.get(name)
for k in ["WANDB_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "GEMINI_API_KEY"]:
    v = secret(k)
    if v:
        os.environ[k] = v
import wandb; wandb.login()
%cd {SM}
""")

md("## Ollama (mô hình mở)")
code("""
!apt-get -qq install -y zstd > /dev/null
!curl -fsSL https://ollama.com/install.sh | sh
# chạy lần lượt từng mô hình (runner model-major) -> chỉ giữ 1 mô hình trong VRAM; context 8k để prompt
# phản hồi tích lũy qua 5 vòng không bị cắt mất phần đầu
os.environ.update(OLLAMA_NUM_PARALLEL=str(WORKERS), OLLAMA_MAX_LOADED_MODELS="1", OLLAMA_KEEP_ALIVE="30m",
                  OLLAMA_CONTEXT_LENGTH="8192")
ollama = subprocess.Popen(["ollama", "serve"], stdout=open(f"{BASE}/ollama.log", "a"), stderr=subprocess.STDOUT)
time.sleep(5)
TAGS = {"qwen3-4b": "qwen3:4b", "qwen3-8b": "qwen3:8b", "qwen3-14b": "qwen3:14b", "gemma3-4b": "gemma3:4b",
        "gemma3-12b": "gemma3:12b", "gemma3-27b": "gemma3:27b", "phi3-mini": "phi3:mini", "phi3-medium": "phi3:medium"}
for m in (OPEN["large"] if not USE_PAPER_MODELS else []):
    !ollama pull {TAGS[m]}
""")

md("## Kiểm tra nhanh")
code("""
!cd {REPO}/repro && python -m pytest -q
""")

md("## S0 — Dữ liệu, mô hình khả dụng, chấm lại đầu ra gốc của tác giả (không tốn API)")
code("""
!python -m smkg.cli download
!python -m smkg.cli check-models --models {' '.join(PAPER + OPEN['large'])}
!python -m smkg.cli score-authors
""")

md("## S1 — EVIA đầy đủ (3 LLM × 5 vòng self-refine)")
code("""
!python -m smkg.cli evia --models {' '.join(MAIN)} --lessons {LESSONS} --run_name {MAIN_RUN} --workers {WORKERS}
""")

md("## S2 — Ablation (tính lại từ log S1)  ·  S4 — lưới K × α  ·  S5a — phân loại lỗi")
code("""
!python -m smkg.cli ablate --run_name {MAIN_RUN} --models {' '.join(MAIN)} --lessons {LESSONS}
!python -m smkg.cli grid   --run_name {MAIN_RUN} --models {' '.join(MAIN)} --lessons {LESSONS}
!python -m smkg.cli errors --run_name {MAIN_RUN} --models {' '.join(MAIN)} --lessons {LESSONS}
""")

md("## S1b — Baseline (dùng mô hình đầu của bộ chính)")
code("""
for b in ["decomposed_qa", "gpt_ner", "self_improving"]:
    !python -m smkg.cli baseline --name {b} --model {MAIN[0]} --lessons {LESSONS} --pool_run {MAIN_RUN}
""")
md("UniversalNER cần vLLM (~13 GB trọng số fp16). Nên chạy ô này ở **một phiên riêng** sau các ô khác vì "
   "`pip install vllm` có thể thay phiên bản torch. Kaggle 2×T4: chia mô hình lên 2 GPU (`--tensor-parallel-size 2`).")
code("""
ollama.terminate(); time.sleep(5)
!pip install -q vllm
TP = "2" if ON_KAGGLE else "1"
uniner = subprocess.Popen(["vllm", "serve", "Universal-NER/UniNER-7B-type", "--port", "8001", "--dtype", "half",
                           "--max-model-len", "4096", "--gpu-memory-utilization", "0.9", "--tensor-parallel-size", TP],
                          stdout=open(f"{BASE}/vllm.log", "a"), stderr=subprocess.STDOUT)
!for i in $(seq 1 120); do curl -s localhost:8001/v1/models > /dev/null && break; sleep 5; done
!python -m smkg.cli baseline --name universal_ner --model uniner-7b --lessons {LESSONS} --pool_run {MAIN_RUN}
uniner.terminate(); time.sleep(5)
ollama = subprocess.Popen(["ollama", "serve"], stdout=open(f"{BASE}/ollama.log", "a"), stderr=subprocess.STDOUT); time.sleep(5)
""")

md("## S3 — Bộ mô hình mở theo kích thước (Bảng 4 của bài)")
code("""
for size in ["small", "mid", "large"]:
    if size == "large" and not USE_PAPER_MODELS:
        continue          # đã chạy ở S1
    for m in OPEN[size]:
        !ollama pull {TAGS[m]}
    !python -m smkg.cli evia --models {' '.join(OPEN[size])} --lessons {LESSONS} --run_name S3-open-{size} --workers {WORKERS}
""")
md("Tùy chọn: Phi-3-small (7B) như bài — không có trên Ollama, phục vụ bằng vLLM ở cổng 8000 rồi chạy lại bộ `mid` với `phi3-small`.")
code("""
# ollama.terminate(); !vllm serve microsoft/Phi-3-small-8k-instruct --trust-remote-code --port 8000 --dtype half &
# !python -m smkg.cli evia --models qwen3-8b gemma3-12b phi3-small --lessons {LESSONS} --run_name S3-open-mid-phi3small
""")

md("## S5 — Căn chỉnh khái niệm–ảnh")
code("""
!python -m smkg.cli align-eval --metrics clip blip --translator {MAIN[0]}
!python -m smkg.cli human-sheet --n 100
# Sau khi hai thành viên điền cột rater_1, rater_2 (0..1) trong human_eval/human_sheet.csv:
# !python -m smkg.cli human-sheet --score {BASE}/out/scimkg/human_eval/human_sheet.csv
""")
code("""
# Tùy chọn: căn chỉnh lại ảnh bằng prompt C1-C4 với mô hình đa phương thức mở (Gemma 3 nhận ảnh)
!python -m smkg.cli regen-align --model gemma3-27b --max_images 200
""")

nb = {"cells": CELLS, "metadata": {"accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]:
    c["source"] = [l + "\n" for l in c["source"].split("\n")]
    c["source"][-1] = c["source"][-1].rstrip("\n")
out = Path(__file__).with_name("SciMKG_Repro.ipynb")
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", out)
