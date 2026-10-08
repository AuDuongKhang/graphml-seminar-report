"""Biến mẫu jsonl thành batch cho LlavaNext; loss chỉ tính trên token câu trả lời (như LLaVA: phần prompt = IGNORE)."""
from PIL import Image

IGNORE = -100
CLIP_MEAN_RGB = tuple(int(x * 255) for x in (0.48145466, 0.4578275, 0.40821073))


def pad_image(img: Image.Image, mode: str = "pad") -> Image.Image:
    """mode=pad: đệm thành hình vuông bằng màu trung bình CLIP (expand2square của LLaVA); anyres: giữ nguyên."""
    img = img.convert("RGB")
    if mode != "pad":
        return img
    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), CLIP_MEAN_RGB)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


def build_messages(prompt: str, has_image: bool, answer: str | None = None) -> list:
    content = ([{"type": "image"}] if has_image else []) + [{"type": "text", "text": prompt}]
    msgs = [{"role": "user", "content": content}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return msgs


def mask_prompt_tokens(input_ids: list, prompt_len: int, attention: list) -> list:
    return [tok if (i >= prompt_len and attention[i]) else IGNORE for i, tok in enumerate(input_ids)]


class Collator:
    """Mỗi mẫu: {"image": path|None, "prompt": str, "answer": str}.

    Một batch phải đồng nhất (toàn có ảnh hoặc toàn không) — các tập dg/dv/text vốn đồng nhất, còn
    huấn luyện joint chỉ trộn dg với dv (đều có ảnh). Cả batch được mã hóa trong một lần gọi processor
    để processor tự đệm số ô ảnh khác nhau ở chế độ anyres. Cắt chuỗi quá max_len ở phần đuôi (token ảnh
    luôn nằm đầu nên không bị cắt).
    """

    def __init__(self, processor, max_len: int = 2048, image_mode: str = "pad"):
        self.proc = processor
        self.max_len = max_len
        self.image_mode = image_mode

    def __call__(self, samples: list) -> dict:
        has_img = [bool(s.get("image")) for s in samples]
        if any(has_img) and not all(has_img):
            raise ValueError("batch trộn mẫu có ảnh và không ảnh; hãy tách theo tập dữ liệu")
        import torch

        imgs = [pad_image(Image.open(s["image"]), self.image_mode) for s in samples] if all(has_img) else None
        prompts = [self.proc.apply_chat_template(build_messages(s["prompt"], imgs is not None),
                                                 add_generation_prompt=True) for s in samples]
        # rstrip: khoảng trắng sau </s> của chat template không được thành token nhãn
        fulls = [self.proc.apply_chat_template(build_messages(s["prompt"], imgs is not None, s["answer"])).rstrip()
                 for s in samples]
        prompt_lens = []
        for i, p in enumerate(prompts):
            kw = {"text": p, "return_tensors": "pt"}   # giữ BOS như lúc suy luận và lúc LLaVA-1.6 tiền huấn luyện
            if imgs is not None:
                kw["images"] = imgs[i]
            prompt_lens.append(int(self.proc(**kw)["attention_mask"][0].sum()))
        self.proc.tokenizer.padding_side = "right"
        kw = {"text": fulls, "return_tensors": "pt", "padding": True}
        if imgs is not None:
            kw["images"] = imgs
        enc = self.proc(**kw)
        ids = enc["input_ids"][:, : self.max_len]
        attn = enc["attention_mask"][:, : self.max_len]
        labels = [mask_prompt_tokens(r.tolist(), pl, a.tolist()) for r, pl, a in zip(ids, prompt_lens, attn)]
        batch = {"input_ids": ids, "attention_mask": attn, "labels": torch.tensor(labels)}
        if imgs is not None:
            batch["pixel_values"] = enc["pixel_values"]
            batch["image_sizes"] = enc["image_sizes"]
        return batch
