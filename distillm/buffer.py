import copy
import random
from collections import deque

import torch


class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.replay_memory = deque(maxlen=args.capacity)
        self.bs = args.batch_size

    def __len__(self):
        return len(self.replay_memory)

    @staticmethod
    def _sample_value(value, index, batch_size):
        if isinstance(value, torch.Tensor):
            if value.ndim == 0 or value.shape[0] != batch_size:
                return value.detach().cpu().clone()
            return value[index].detach().cpu().clone()
        if isinstance(value, (list, tuple)) and len(value) == batch_size:
            return copy.deepcopy(value[index])
        return copy.deepcopy(value)

    @staticmethod
    def _collate(records):
        if not records:
            return None
        batch = {}
        for key in records[0]:
            values = [record[key] for record in records]
            if all(isinstance(value, torch.Tensor) for value in values):
                batch[key] = torch.stack(values, dim=0)
            else:
                batch[key] = values
        return batch

    def sample(self, sample_size=None):
        sample_size = self.bs if sample_size is None else sample_size
        if sample_size > len(self.replay_memory):
            raise ValueError(
                f"cannot sample {sample_size} examples from replay buffer of size "
                f"{len(self.replay_memory)}"
            )
        records = random.sample(self.replay_memory, k=sample_size)
        model_data = self._collate([record["model_data"] for record in records])
        no_model_data = self._collate([record["no_model_data"] for record in records])
        gen_records = [record["gen_data"] for record in records]
        gen_data = None if all(record is None for record in gen_records) else self._collate(gen_records)
        return model_data, no_model_data, gen_data

    def move_to_device(self, model_data, no_model_data, gen_data, device):
        for batch in (model_data, no_model_data, gen_data):
            if batch is None:
                continue
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)

        return model_data, no_model_data, gen_data

    def move_to_memory(self, model_data, no_model_data, gen_data=None):
        batch_size = model_data["input_ids"].shape[0]
        for index in range(batch_size):
            record = {}
            for name, batch in (
                ("model_data", model_data),
                ("no_model_data", no_model_data),
                ("gen_data", gen_data),
            ):
                if batch is None:
                    record[name] = None
                    continue
                record[name] = {
                    key: self._sample_value(value, index, batch_size)
                    for key, value in batch.items()
                }
            self.replay_memory.append(record)