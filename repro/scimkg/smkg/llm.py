"""Client LLM kiểu OpenAI dùng chung cho GPT-4o / DeepSeek / Gemini / Ollama / vLLM.

- Cache bền trên đĩa (SQLite) theo (model, prompt, tham số): chạy lại hay ablation không gọi lại API.
- Không bao giờ ném lỗi mạng/API ra ngoài: trả LLMReply(text=None, error=...) và đếm vào `counters`
  (dùng cho phân loại lỗi NR của Fig 5). Lỗi không được cache.
"""
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "models.yaml"
# Bảng 6 của bài
PAPER_PARAMS = {"temperature": 0.0, "top_p": 0.1, "frequency_penalty": 0.5, "max_tokens": 1000, "seed": 42}
_BULLET = re.compile(r"^\s*(?:[-•*·]|\d+[.)])\s*(.*?)\s*$")
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass
class LLMReply:
    text: str | None
    error: str | None
    cached: bool = False


def load_model_configs(path: Path = CONFIG) -> dict:
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def strip_think(text: str) -> str:
    """Bỏ khối <think> của Qwen3. Khối mở mà không đóng (bị cắt bởi max_tokens) -> rỗng, tính là không phản hồi."""
    text = text or ""
    if "<think>" in text and "</think>" not in text:
        return ""
    return _THINK.sub("", text).strip()


def parse_concepts(text: str | None):
    """Danh sách `-xxx` -> (khái niệm, trạng thái ok | format_error | no_response)."""
    if text is None or not text.strip():
        return [], "no_response"
    items = []
    for line in strip_think(text).splitlines():
        m = _BULLET.match(line)
        if m and m.group(1).strip():
            items.append(m.group(1).strip())
    return (items, "ok") if items else ([], "format_error")


class LLMClient:
    def __init__(self, name: str, cfg: dict, cache_dir: Path, params: dict | None = None, sleep=time.sleep,
                 retries: int = 3, timeout: float = 120.0):
        self.name, self.cfg = name, cfg
        self.model = cfg.get("model", name)
        self.params = dict(PAPER_PARAMS if params is None else params)
        self.params.update(cfg.get("params") or {})          # tham số riêng theo mô hình (vd max_tokens của UniNER)
        for p in cfg.get("drop_params", []) or []:
            self.params.pop(p, None)
        self.sleep, self.retries, self.timeout = sleep, retries, timeout
        self.counters = Counter()
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(Path(cache_dir) / "llm_cache.sqlite"), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
        self._lock = threading.Lock()
        self._client = None

    def _key(self, prompt: str, params: dict, image_digest: str = "") -> str:
        raw = json.dumps([self.model, prompt, params, image_digest], ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _call(self, prompt: str, image=None, **params) -> str:
        if self._client is None:
            from openai import OpenAI
            key_env = self.cfg.get("key_env")
            self._client = OpenAI(base_url=self.cfg.get("base_url"), timeout=self.timeout,
                                  api_key=os.environ.get(key_env, "") if key_env else "not-needed")
        if self.cfg.get("mode") == "completion":   # mô hình không có chat template (UniNER)
            r = self._client.completions.create(model=self.model, prompt=prompt, **params)
            return r.choices[0].text
        content = prompt
        if image is not None:
            import base64
            b64 = base64.b64encode(Path(image).read_bytes()).decode()
            content = [{"type": "text", "text": prompt},
                       {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]
        r = self._client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": content}],
                                                 **params)
        return r.choices[0].message.content

    def chat(self, prompt: str, image=None, **override) -> LLMReply:
        """image: đường dẫn ảnh (cho MLLM, vd căn chỉnh khái niệm-ảnh); được đưa vào khóa cache."""
        params = {**self.params, **override}
        if self.cfg.get("no_think"):
            prompt = prompt + " /no_think"                     # tắt chế độ suy nghĩ của Qwen3
        digest = hashlib.sha256(Path(image).read_bytes()).hexdigest() if image is not None else ""
        key = self._key(prompt, params, digest)
        with self._lock:
            row = self._db.execute("SELECT v FROM c WHERE k=?", (key,)).fetchone()
        if row:
            self.counters["cache_hit"] += 1
            return LLMReply(json.loads(row[0]), None, cached=True)
        err = None
        for attempt in range(self.retries):
            try:
                text = self._call(prompt, image=image, **params) if image is not None else self._call(prompt, **params)
                text = strip_think(text) if text else text
                if not text or not text.strip():
                    err = "no_response"
                    break
                with self._lock:
                    self._db.execute("INSERT OR REPLACE INTO c VALUES (?, ?)", (key, json.dumps(text, ensure_ascii=False)))
                    self._db.commit()
                self.counters["ok"] += 1
                return LLMReply(text, None)
            except TimeoutError:
                err = "timeout"
            except Exception as e:  # lỗi mạng/API bất kỳ: thử lại rồi ghi nhận, không ném ra ngoài
                err = "timeout" if "timeout" in type(e).__name__.lower() else "api_error"
            if attempt < self.retries - 1:
                self.sleep(2 ** attempt)
        self.counters[err] += 1
        return LLMReply(None, err)


def make_clients(names: list, cache_dir: Path, params: dict | None = None) -> dict:
    cfgs = load_model_configs()
    return {n: LLMClient(n, cfgs[n], Path(cache_dir) / n, params) for n in names}


def check_models(names: list, cache_dir: Path) -> dict:
    """Thử gọi từng mô hình; nếu lỗi và có `fallback` thì thử mô hình thay thế. Trả trạng thái để ghi DEVIATIONS."""
    cfgs = load_model_configs()
    out = {}
    for n in names:
        cfg = cfgs[n]
        key_env = cfg.get("key_env")
        if key_env and not os.environ.get(key_env):
            out[n] = f"skip: thiếu {key_env}"
            continue
        r = LLMClient(n, cfg, Path(cache_dir) / "_check", {"max_tokens": 5}).chat(f"Reply with OK. ({time.time()})")
        if r.text:
            out[n] = f"ok ({cfg['model']})"
        elif cfg.get("fallback"):
            fb = dict(cfg, model=cfg["fallback"])
            r2 = LLMClient(n, fb, Path(cache_dir) / "_check", {"max_tokens": 5}).chat(f"Reply with OK. ({time.time()})")
            out[n] = f"fallback {cfg['fallback']}" if r2.text else f"lỗi: {r.error}"
        else:
            out[n] = f"lỗi: {r.error}"
    return out
