"""Sinh GraphVis_Repro.ipynb. Sửa file này rồi chạy lại thay vì sửa trực tiếp notebook:
    python repro/graphvis/notebooks/make_notebook.py
"""
import json
from pathlib import Path

CELLS = []


def md(s):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md("""
# Tái lập GraphVis — Kaggle / Colab (GPU miễn phí)

Chạy **từ trên xuống**. Mỗi ô thực nghiệm có thể chạy lại sau khi phiên bị ngắt: dữ liệu đã dựng và
giai đoạn đã huấn luyện xong được bỏ qua, giai đoạn dở dang resume từ checkpoint.

| Ô | Thực nghiệm | Mục báo cáo |
|---|---|---|
| G0 | dựng dữ liệu (đồ thị con, ảnh, D_g, D_v) | VII-A |
| G1a | mô hình gốc: text / image / KAPING trên CSQA, OBQA | VII-B |
| G1b, G1c | GraphVis curriculum trên CSQA, OBQA | VII-B |
| G6 | 6 tác vụ đọc ảnh trước/sau giai đoạn 1 | VII-E |
| G2 | ScienceQA-IMG, MMBench, POPE | VII-C |
| G4 | joint / graph-first / OCR-first | VII-D |
| G3 | ScienceQA cho bản joint | VII-C |
| G5 | kết hợp KAPING | VII-D |
| G7 | ảnh 50×50 | VII-E |
| C3 | đối chứng tinh chỉnh không ảnh | VI-C |
| C1 | độ chính xác theo số nút | VI-C |
| C2 | độ chính xác theo độ phân giải | VI-C |

**Trước khi chạy:** Kaggle → *Settings*: bật Internet, chọn GPU (T4 x2 hoặc P100), và bật
*Persistence → Files only* (nếu không, `/kaggle/working` mất khi hết phiên). Thêm *Secret* `WANDB_API_KEY`
(và `HF_TOKEN` nếu dùng `HUB_REPO`; `GITHUB_TOKEN` nếu repo còn private — fine-grained token, chỉ quyền
*Contents: Read-only* trên repo này). Colab → *Runtime type* GPU; thêm secret cùng tên trong mục 🔑.
Khuyến nghị đặt `HUB_REPO`: adapter của mỗi giai đoạn được đẩy lên HF Hub và tự tải lại ở phiên sau / máy khác.
""")

code("""
# ==== CẤU HÌNH ====
SCALE = "lite"          # tiny (thử nhanh) | lite (mặc định, ~5-7 giờ/lượt train trên T4) | full (như bài)
QUANT, DTYPE = "4bit", "fp16"   # A100/L4: "none", "bf16"
WANDB_PROJECT = "graphml-seminar-report"
WANDB_ENTITY = ""       # để trống = tài khoản mặc định
HUB_REPO = ""           # vd "your-hf-name/graphvis-adapters" để lưu adapter lên HF Hub (chuyển giữa Kaggle/Colab)
REPO_URL = "https://github.com/AuDuongKhang/graphml-seminar-report.git"
BRANCH = "main"
G6_MAX = {"tiny": 60, "lite": 600, "full": None}[SCALE]   # số câu đọc ảnh (6 câu/ảnh); 600 ~ 100 ảnh
VQA_MAX = {"tiny": {"scienceqa": 50, "mmbench": 20, "pope": 50},
           "lite": {"scienceqa": None, "mmbench": 400, "pope": 500},
           "full": {"scienceqa": None, "mmbench": None, "pope": None}}[SCALE]
""")

code("""
import os, sys, subprocess
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # QLoRA chạy 1 GPU; tránh Trainer bọc DataParallel trên Kaggle 2xT4
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
if WANDB_ENTITY:
    os.environ["WANDB_ENTITY"] = WANDB_ENTITY
if HUB_REPO:
    os.environ["GRAPHML_HUB_REPO"] = HUB_REPO      # eval tự tải adapter còn thiếu từ HF Hub
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
    return os.environ.get(name)

def git(*args):
    # repo private: dùng secret GITHUB_TOKEN qua header HTTP, không ghi token vào URL hay .git/config
    cmd = ["git"]
    token = secret("GITHUB_TOKEN")
    if token:
        import base64
        basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        cmd += ["-c", f"http.extraheader=AUTHORIZATION: basic {basic}"]
    subprocess.run(cmd + list(args), check=True)

REPO = f"{BASE}/repo"
if not os.path.exists(REPO):
    git("clone", "--depth", "1", "-b", BRANCH, REPO_URL, REPO)
else:
    git("-C", REPO, "pull")
GV = f"{REPO}/repro/graphvis"
OUT = os.environ["GRAPHML_OUT"] + "/graphvis"
print("BASE =", BASE)
""")

code("""
!apt-get -qq install -y graphviz > /dev/null
!pip install -q -r {GV}/requirements.txt
""")

code("""
os.environ["WANDB_API_KEY"] = secret("WANDB_API_KEY") or ""
if HUB_REPO:
    os.environ["HF_TOKEN"] = secret("HF_TOKEN") or ""
import wandb; wandb.login()
M = f"--quant {QUANT} --dtype {DTYPE}"
S = f"--set quant={QUANT} dtype={DTYPE}" + (f" hub_repo={HUB_REPO}" if HUB_REPO else "")
def adapter(exp, stage):
    return f"{OUT}/{exp}/{stage}/final"
%cd {GV}
""")

md("## Kiểm tra nhanh (chạy 1 lần): test CPU + test GPU + dry-run huấn luyện")
code("""
!cd {REPO}/repro && python -m pytest -q
!cd {REPO}/repro && python -m pytest -q -m gpu graphvis/tests/test_collate.py -p no:cacheprovider
!python -m gv.cli build-data --dataset csqa --split train --scale tiny --tag smoke --no_wandb
!python -m gv.cli train --config configs/G1_csqa_curriculum.yaml --dry_run {S} data_tag=smoke
""")

md("## G0 — Dựng dữ liệu")
code("""
for ds, sp in [("csqa", "train"), ("csqa", "dev"), ("obqa", "train"), ("obqa", "test")]:
    !python -m gv.cli build-data --dataset {ds} --split {sp} --scale {SCALE}
""")

md("## G1a — Mô hình gốc (không tinh chỉnh): text / image / KAPING")
code("""
for ds, sp in [("csqa", "dev"), ("obqa", "test")]:
    for mode in ["text", "image", "kaping"]:
        !python -m gv.cli eval-mcqa --dataset {ds} --split {sp} --mode {mode} --exp_id G1a-base-{ds}-{mode} --group G1 {M}
""")

md("## G1b — GraphVis curriculum trên CSQA (mô hình chính)")
code("""
!python -m gv.cli train --config configs/G1_csqa_curriculum.yaml {S}
!python -m gv.cli eval-mcqa --dataset csqa --split dev --mode image --adapter {adapter('G1-csqa-curriculum','stage2_qa')} --exp_id G1b-graphvis-csqa-image --group G1 {M}
""")

md("## G1c — GraphVis curriculum trên OpenBookQA")
code("""
!python -m gv.cli train --config configs/G1_obqa_curriculum.yaml {S}
!python -m gv.cli eval-mcqa --dataset obqa --split test --mode image --adapter {adapter('G1-obqa-curriculum','stage2_qa')} --exp_id G1c-graphvis-obqa-image --group G1 {M}
""")

md("## G6 — Đọc ảnh đồ thị: mô hình gốc vs sau giai đoạn 1")
code("""
GM = f"--max_eval {G6_MAX}" if G6_MAX else ""
!python -m gv.cli eval-graph-reading --exp_id G6-base --group G6 --batch_size 4 {GM} {M}
!python -m gv.cli eval-graph-reading --adapter {adapter('G1-csqa-curriculum','stage1_graph_reading')} --exp_id G6-after-stage1 --group G6 --batch_size 4 {GM} {M}
""")

md("## G2 — VQA zero-shot: mô hình gốc vs GraphVis (CSQA)")
code("""
for name, ad in [("base", "none"), ("graphvis", adapter('G1-csqa-curriculum','stage2_qa'))]:
    for b in ["scienceqa", "mmbench", "pope"]:
        mx = f"--max_eval {VQA_MAX[b]}" if VQA_MAX[b] else ""
        !python -m gv.cli eval-vqa --benchmark {b} --adapter {ad} --exp_id G2-{name}-{b} --group G2 {mx} {M}
""")

md("## G4 — Ablation thứ tự huấn luyện (Fig 7)")
code("""
for cfg, exp, stage in [("G4_joint", "G4-csqa-joint", "joint"),
                        ("G4_graph_first", "G4-csqa-graph-first", "stage2_qa"),
                        ("G4_ocr_first", "G4-csqa-ocr-first", "stage2_qa")]:
    !python -m gv.cli train --config configs/{cfg}.yaml {S}
    !python -m gv.cli eval-mcqa --dataset csqa --split dev --mode image --adapter {adapter(exp, stage)} --exp_id {exp}-eval --group G4 {M}
""")

md("## G3 — ScienceQA cho bản joint (Table 6)")
code("""
!python -m gv.cli eval-vqa --benchmark scienceqa --adapter {adapter('G4-csqa-joint','joint')} --exp_id G3-joint-scienceqa --group G3 {M}
""")

md("## G5 — Kết hợp KAPING (Table 3)")
code("""
for name, ad in [("base", "none"), ("joint", adapter('G4-csqa-joint','joint')),
                 ("curriculum", adapter('G1-csqa-curriculum','stage2_qa'))]:
    !python -m gv.cli eval-mcqa --dataset csqa --split dev --mode image+kaping --adapter {ad} --exp_id G5-{name}-kaping --group G5 {M}
""")

md("## G7 — Ảnh đồ thị 50×50 (Fig 9)")
code("""
for sp in ["train", "dev"]:
    !python -m gv.cli build-data --dataset csqa --split {sp} --scale {SCALE} --image_size 50 --tag lowres50
!python -m gv.cli train --config configs/G7_lowres.yaml {S}
!python -m gv.cli eval-mcqa --dataset csqa --split dev --mode image --data_tag lowres50 --adapter {adapter('G7-csqa-lowres50','stage2_qa')} --exp_id G7-lowres50-eval --group G7 {M}
""")

md("## C3 — Đối chứng: tinh chỉnh không ảnh")
code("""
!python -m gv.cli train --config configs/C3_text_only.yaml {S}
!python -m gv.cli eval-mcqa --dataset csqa --split dev --mode text --adapter {adapter('C3-csqa-text-only','text_qa')} --exp_id C3-text-only-eval --group C3 {M}
""")

md("## C1 — Độ chính xác theo số nút (dùng lại dự đoán đã lưu)")
code("""
!python -m gv.cli c1-report --eval_exp_ids G1a-base-csqa-text G1a-base-csqa-image G1b-graphvis-csqa-image C3-text-only-eval G7-lowres50-eval
""")

md("## C2 — Độ chính xác theo độ phân giải ảnh")
code("""
!python -m gv.cli resolution-sweep --exp_prefix C2-base {M}
!python -m gv.cli resolution-sweep --adapter {adapter('G1-csqa-curriculum','stage2_qa')} --exp_prefix C2-graphvis {M}
""")

nb = {"cells": CELLS, "metadata": {"accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]:
    c["source"] = [l + "\n" for l in c["source"].split("\n")]
    c["source"][-1] = c["source"][-1].rstrip("\n")
out = Path(__file__).with_name("GraphVis_Repro.ipynb")
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", out)
