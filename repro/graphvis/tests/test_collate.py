import pytest
from PIL import Image

from gv.model.collate import IGNORE, build_messages, mask_prompt_tokens, pad_image


def test_mask_prompt_tokens_masks_prompt_and_padding():
    ids = [1, 10, 11, 12, 13, 20, 21, 2, 0]
    labels = mask_prompt_tokens(ids, prompt_len=5, attention=[1] * 8 + [0])
    assert labels == [IGNORE] * 5 + [20, 21, 2] + [IGNORE]


def test_mask_prompt_tokens_all_prompt_when_truncated():
    assert mask_prompt_tokens([1, 2, 3], prompt_len=5, attention=[1, 1, 1]) == [IGNORE] * 3


def test_build_messages_with_and_without_image():
    m = build_messages("Q?", has_image=True, answer="A")
    assert m[0]["content"][0] == {"type": "image"} and m[0]["content"][1] == {"type": "text", "text": "Q?"}
    assert m[1] == {"role": "assistant", "content": [{"type": "text", "text": "A"}]}
    m2 = build_messages("Q?", has_image=False)
    assert len(m2) == 1 and m2[0]["content"] == [{"type": "text", "text": "Q?"}]


def test_pad_image_square_mean_color():
    out = pad_image(Image.new("RGB", (40, 10), "white"), mode="pad")
    assert out.size == (40, 40)
    assert out.getpixel((0, 0)) == (122, 116, 104)       # màu trung bình CLIP
    assert pad_image(Image.new("RGB", (40, 10)), mode="anyres").size == (40, 10)


def test_collator_rejects_mixed_image_and_text_batches():
    from gv.model.collate import Collator

    with pytest.raises(ValueError):
        Collator(processor=None)([{"image": "a.png", "prompt": "q", "answer": "a"},
                                  {"image": None, "prompt": "q", "answer": "a"}])


@pytest.mark.gpu
def test_model_forward_backward_on_gpu(tmp_path):
    """Chạy trên Colab/Kaggle: python -m pytest -m gpu graphvis/tests/test_collate.py"""
    import torch

    from gv.model.collate import Collator
    from gv.model.load import load_model_and_processor

    model, proc = load_model_and_processor(quant="4bit", dtype="fp16", lora={"r": 128, "alpha": 256})
    img = tmp_path / "x.png"
    Image.new("RGB", (300, 120), "white").save(img)
    batch = Collator(proc)([{"image": str(img), "prompt": "How many nodes?", "answer": "There are 2 nodes."},
                            {"image": str(img), "prompt": "Question: 1+1? A. 2 B. 3", "answer": "A"}])
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(**{k: v.to(model.device) for k, v in batch.items()})
    assert torch.isfinite(out.loss)
    out.loss.backward()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 0.3e9 < trainable < 0.4e9
