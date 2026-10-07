"""Sinh câu trả lời tham lam theo batch cho LlavaNext (mọi đánh giá đều do_sample=False như bài)."""
from PIL import Image


def load_image(src, image_mode: str = "pad", image_size: int | None = None):
    from ..data.render import pad_to_square_resize
    from ..model.collate import pad_image

    img = src if isinstance(src, Image.Image) else Image.open(src)
    if image_size:                       # thí nghiệm độ phân giải: thu nhỏ trước, processor sẽ phóng lại 336
        img = pad_to_square_resize(img, image_size)
    return pad_image(img, image_mode)


def generate(model, proc, rows: list, image_mode: str = "pad", batch_size: int = 8, max_new_tokens: int = 16,
             dtype: str = "fp16", image_size: int | None = None, progress: bool = True) -> list:
    import torch

    from ..model.collate import build_messages
    from ..model.load import torch_dtype

    dt = torch_dtype(dtype)
    proc.tokenizer.padding_side = "left"
    model.eval()
    outs = []
    it = range(0, len(rows), batch_size)
    if progress:
        from tqdm.auto import tqdm
        it = tqdm(it, desc="generate")
    for i in it:
        batch = rows[i:i + batch_size]
        with_img = [r.get("image") is not None for r in batch]
        if any(with_img) and not all(with_img):
            raise ValueError("batch trộn mẫu có ảnh và không ảnh")
        imgs = [load_image(r["image"], image_mode, image_size) for r in batch] if all(with_img) else None
        prompts = [proc.apply_chat_template(build_messages(r["prompt"], imgs is not None), add_generation_prompt=True)
                   for r in batch]
        kw = {"text": prompts, "padding": True, "return_tensors": "pt"}
        if imgs is not None:
            kw["images"] = imgs
        enc = proc(**kw).to(model.device)
        if "pixel_values" in enc:
            enc["pixel_values"] = enc["pixel_values"].to(dt)
        with torch.inference_mode(), torch.autocast("cuda", dtype=dt, enabled=torch.cuda.is_available()):
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=proc.tokenizer.pad_token_id)
        outs += proc.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return [o.strip() for o in outs]
