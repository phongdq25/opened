import torch

import ced_losses as new
import reference_ced_loss as ref
from conftest import ced_args

KD_TYPES = ["kd", "fkl", "rkl", "sfkl", "srkl", "jsd", "tvd", "csd", "amid", "no"]


def _logits(seed):
    return torch.randn(3, 9, 64, generator=torch.Generator().manual_seed(seed))


def test_distil_losses_unchanged_by_the_move():
    s, t = _logits(0), _logits(1)
    label = torch.randint(0, 64, (3, 9), generator=torch.Generator().manual_seed(2))
    label[0, :4] = -100
    for kd_type in KD_TYPES:
        for div in ("fkl", "ab"):
            args = ced_args(type=kd_type, amid_div_name=div)
            assert torch.equal(new.get_distil_loss(args, t, {"label": label}, s),
                               ref.get_distil_loss(args, t, {"label": label}, s)), kd_type


def test_span_loss_unchanged_by_the_move():
    g = torch.Generator().manual_seed(3)
    hid_s = [None] + [torch.randn(2, 12, 16, generator=g) for _ in range(4)]
    hid_t = [torch.randn(2, 12, 16, generator=g) for _ in range(5)]
    mask = torch.ones(2, 12, dtype=torch.long)
    mask[1, 9:] = 0
    offsets = torch.stack([torch.arange(12) * 4, torch.arange(12) * 4 + 3], dim=-1).unsqueeze(0).repeat(2, 1, 1)
    spans = [[(0, 7), (16, 27), (32, 40), (41, 44)], [(4, 11), (20, 30)]]
    for metric in ("cosine", "dot", "l2", "cka"):
        args = ced_args(span_metric=metric)
        assert torch.equal(new.compute_overall_span_loss(mask, hid_s, hid_t, offsets, spans, args),
                           ref.compute_overall_span_loss(mask, hid_s, hid_t, offsets, spans, args)), metric


def test_label_window_spans_every_labelled_column():
    label = torch.full((2, 10), -100)
    label[0, 3:5] = 1
    label[1, 6] = 2
    assert new.label_window(label) == (3, 7)


def test_label_window_of_an_unlabelled_batch_is_one_column():
    assert new.label_window(torch.full((2, 4), -100)) == (0, 1)
