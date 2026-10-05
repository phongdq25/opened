import random

import torch

from conftest import ced_args, need, tiny_qwen
from data_utils.lm_datasets import LMTrainDataset

FEWREL = "processed_data/fewrel_perm0/1/qwen/"
ROWS = [0, 5, 17, 40, 333, 1200, 2500, 3399]


def load(tokenizer, dynamic):
    return LMTrainDataset(ced_args(dynamic_pad_effective=dynamic), tokenizer, need(FEWREL), "train", -1, 1,
                          random.Random(0))


def test_width_is_the_longest_row_rounded_up_to_64(tokenizer):
    ds = load(tokenizer, True)
    mb, _, _, _, _ = ds.collate([ds[i] for i in ROWS])
    longest = max(len(ds[i]["input_ids"]) - 2 for i in ROWS)        # minus the sentinel and the shift
    assert mb["input_ids"].shape[1] == min(768, -(-longest // 64) * 64)


def test_dynamic_batch_equals_the_fixed_batch_on_its_columns(tokenizer):
    fixed, dyn = load(tokenizer, False), load(tokenizer, True)
    a = fixed.collate([fixed[i] for i in ROWS])
    b = dyn.collate([dyn[i] for i in ROWS])
    w = b[0]["input_ids"].shape[1]
    for key in ("input_ids", "attention_mask"):
        assert torch.equal(a[0][key][:, :w], b[0][key])
    assert torch.equal(a[1]["label"][:, :w], b[1]["label"])
    assert (a[1]["label"][:, w:] == -100).all() and (a[0]["attention_mask"][:, w:] == 0).all()
    assert torch.equal(a[2]["input_ids"], b[2]["input_ids"])          # generation prompts keep their width


def test_dynamic_padding_leaves_every_real_logit_unchanged(tokenizer):
    fixed, dyn = load(tokenizer, False), load(tokenizer, True)
    a = fixed.collate([fixed[i] for i in ROWS])
    b = dyn.collate([dyn[i] for i in ROWS])
    model = tiny_qwen(0).eval()
    with torch.no_grad():
        la = model(**a[0], use_cache=False).logits
        lb = model(**b[0], use_cache=False).logits
    real = b[0]["attention_mask"].bool()
    torch.testing.assert_close(la[:, :lb.shape[1]][real], lb[real], rtol=1e-5, atol=1e-5)
