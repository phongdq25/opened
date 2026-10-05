import random

from conftest import ced_args, need
from data_utils.lm_datasets import LMTrainDataset, resolve_num


def test_resolve_num_caps_at_the_split_size():
    assert resolve_num(128, 38) == 38


def test_resolve_num_keeps_a_smaller_cap_and_minus_one_means_all():
    assert resolve_num(16, 38) == 16
    assert resolve_num(-1, 38) == 38


def test_dev_num_larger_than_the_split_does_not_hang(tokenizer):
    # ACE task0 dev has 38 rows; asking for 128 used to loop forever in DistributedMMapIndexedDataset
    ds = LMTrainDataset(ced_args(), tokenizer, need("processed_data/ace_b10_perm0/0/qwen/"), "valid", 128, 1,
                        random.Random(0))
    assert len(ds) == 38
    ds.collate([ds[i] for i in range(len(ds))])
