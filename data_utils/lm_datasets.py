import random
import torch
import os
import json


def balanced_smoke_indices(replay_flags, limit):
    if limit <= 0:
        return None
    if not replay_flags or not any(replay_flags):
        return list(range(min(limit, len(replay_flags))))
    replay = [index for index, flag in enumerate(replay_flags) if flag]
    current = [index for index, flag in enumerate(replay_flags) if not flag]
    replay_limit = min(len(replay), limit // 2)
    current_limit = min(len(current), limit - replay_limit)
    remaining = limit - replay_limit - current_limit
    if remaining > 0:
        extra_replay = min(len(replay) - replay_limit, remaining)
        replay_limit += extra_replay
        remaining -= extra_replay
        current_limit += min(len(current) - current_limit, remaining)
    selected = current[:current_limit] + replay[:replay_limit]
    return selected
import pickle
import numpy as np
from torch.utils.data import Dataset
from .distributed_indexed import DistributedMMapIndexedDataset

from torch.distributed import get_rank, get_world_size, barrier
from utils import print_rank
from utils import save_rank


class LMTrainDataset(Dataset):
    def __init__(self, args, tokenizer, path, split, num, ratio, rng_sample: random.Random):
        self.args = args
        self.tokenizer = tokenizer
        self.split = split
        self.pad_id = self.tokenizer.eos_token_id
        self.ratio = ratio
        self.max_length = args.max_length
        self.max_prompt_length = args.max_prompt_length
        self.rng_sample = rng_sample
        self.lm_ctx = DistributedMMapIndexedDataset(path, f"{split}", get_rank(), get_world_size())
        self.t_lm_ctx = None

        if os.path.exists(os.path.join(path, f"teacher_train_0.bin")) and split == "train":
            self.t_lm_ctx = DistributedMMapIndexedDataset(path, f"teacher_train", get_rank(), get_world_size())

        if os.path.exists(os.path.join(path, f"{split}.jsonl")):
            with open(os.path.join(path, f"{split}.jsonl")) as f:
                self.raw = [json.loads(line) for line in f.readlines()]
                self.answers = [x["response"] if isinstance(x["response"], list) else [x["response"]] for x in self.raw]
                self.full_texts = [x["prompt"] + x["response"] for x in self.raw]
                self.offset_mapping = [tokenizer(text, return_offsets_mapping=True, truncation=True, 
                                                 max_length=self.max_length, padding="max_length",
                                                 add_special_tokens=False, return_tensors="pt")["offset_mapping"]
                                       for text in self.full_texts]
                
                self.get_span_offsets()
                self.get_replay_flags()

        self.sample_indices = None
        smoke_rows = getattr(args, "ced_smoke_rows", 0)
        if split == "train" and smoke_rows > 0:
            flags = self.replay_flags or [False] * len(self.lm_ctx)
            self.sample_indices = balanced_smoke_indices(flags, smoke_rows)
            self.num = len(self.sample_indices)
            print_rank(
                f"CED smoke subset: {self.num} rows "
                f"({sum(flags[index] for index in self.sample_indices)} replay)"
            )
        elif num == -1:
            self.num = len(self.lm_ctx)
        else:
            self.num = num

        print_rank(f"Num LM instances: {len(self.lm_ctx)}")

    def get_span_offsets(self):
        self.span_offsets = []
        for item, full_text in zip(self.raw, self.full_texts):
            response_str = item["response"]
            response_json = json.loads(response_str)

            values_to_find = []
            for event in response_json.get("events", []):
                if not isinstance(event, list) or len(event) < 2:
                    continue
                values_to_find.append(event[0])  # 1. trigger
                values_to_find.append(event[1])  # 2. event_type

                if len(event) > 3:
                    for arg in event[2]:             # 3. Duyệt qua các arguments
                        if not isinstance(arg, list) or len(arg) < 2:
                            continue  # malformed (e.g. teacher-generated) argument
                        values_to_find.append(arg[0])  # arg_span
                        values_to_find.append(arg[1])  # arg_role

                    values_to_find.append(event[3])  # 4. description

                else:
                    values_to_find.append(event[2])  # 3. description

            result_tuples = []
            # search_start_idx = len(full_text) - len(response_str)
            search_start_idx = 0

            for val in values_to_find:
                search_str = f'{val}'
                char_start = full_text.find(search_str, search_start_idx)
                
                if char_start != -1:
                    char_end = char_start + len(val)
                    result_tuples.append((char_start, char_end))
                    search_start_idx = char_end + 1

            self.span_offsets.append(result_tuples)

    def get_replay_flags(self):
        # CED: a sample is a replay exemplar when all its event types belong to
        # tasks before --ced-task-id (task order read from --ced-streams-file).
        self.replay_flags = None
        streams_file = getattr(self.args, "ced_streams_file", None)
        task_id = getattr(self.args, "ced_task_id", None)
        if streams_file is None or task_id is None or self.split != "train":
            return
        with open(streams_file) as f:
            streams = json.load(f)
        old_types = set()
        for s in streams[:task_id]:
            old_types.update(s)
        flags = []
        for item in self.raw:
            events = json.loads(item["response"]).get("events", [])
            types = {e[1] for e in events}
            flags.append(bool(types) and types <= old_types)
        self.replay_flags = flags
        print_rank(f"CED replay flags: {sum(flags)}/{len(flags)} replay samples (task {task_id})")

        # F3 (kd-scope=pl): char spans of OLD-type event tuples in each row's
        # full_text — non-replay rows with such events are pseudo-labeled; KD is
        # restricted to these token positions where the old teacher is reliable.
        def _spans_for(events, want_old, full_text):
            # collect char spans of event values whose type is old (want_old=True)
            # or new (want_old=False), in event order with a moving cursor.
            values = []
            for e in events:
                if (e[1] in old_types) != want_old:
                    continue
                values.append(e[0])
                values.append(e[1])
                if len(e) > 3:
                    for arg in e[2]:
                        values.append(arg[0])
                        values.append(arg[1])
                    values.append(e[3])
                else:
                    values.append(e[2])
            spans = []
            search_start = 0
            for val in values:
                cs = full_text.find(f'{val}', search_start)
                if cs != -1:
                    spans.append((cs, cs + len(val)))
                    search_start = cs + len(val) + 1
            return spans

        self.old_span_offsets = []
        self.new_span_offsets = []   # LwF: new-type event spans on non-replay rows
        n_pl = 0
        for item, full_text, flag in zip(self.raw, self.full_texts, flags):
            old_spans, new_spans = [], []
            if not flag:  # replay rows already get full-response KD
                events = json.loads(item["response"]).get("events", [])
                old_spans = _spans_for(events, True, full_text)
                new_spans = _spans_for(events, False, full_text)
                if old_spans:
                    n_pl += 1
            self.old_span_offsets.append(old_spans)
            self.new_span_offsets.append(new_spans)
        n_new = sum(1 for s in self.new_span_offsets if s)
        print_rank(f"CED pl rows with old-event spans: {n_pl}/{len(flags)}")
        print_rank(f"CED rows with new-event spans (LwF mask): {n_new}/{len(flags)}")

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        return self._get_lm(index)

    def _get_lm(self, index):
        if self.sample_indices is not None:
            index = self.sample_indices[index]
        data = self.lm_ctx[index]
        input_ids = data.astype(int)

        t_input_ids = None
        if self.t_lm_ctx is not None:
            t_data = self.t_lm_ctx[index]
            t_input_ids = t_data.astype(int)

        is_replay = bool(self.replay_flags[index]) if getattr(self, "replay_flags", None) else False

        # F3: label-position mask for old-event tokens of pseudo-labeled rows.
        # label[k] predicts token k+1, so shift the token-level mask left by one.
        old_token_mask = torch.zeros(self.max_length, dtype=torch.bool)
        old_spans = getattr(self, "old_span_offsets", None)
        if old_spans is not None and old_spans[index]:
            om = self.offset_mapping[index][0]  # (max_length, 2)
            tok_in = torch.zeros(self.max_length, dtype=torch.bool)
            for cs, ce in old_spans[index]:
                tok_in |= (om[:, 0] < ce) & (om[:, 1] > cs) & (om[:, 1] > om[:, 0])
            old_token_mask[:-1] = tok_in[1:]

        # LwF: label-position mask for NEW-type event tokens (excluded from LwF KD).
        new_token_mask = torch.zeros(self.max_length, dtype=torch.bool)
        new_spans = getattr(self, "new_span_offsets", None)
        if new_spans is not None and new_spans[index]:
            om = self.offset_mapping[index][0]
            tok_in = torch.zeros(self.max_length, dtype=torch.bool)
            for cs, ce in new_spans[index]:
                tok_in |= (om[:, 0] < ce) & (om[:, 1] > cs) & (om[:, 1] > om[:, 0])
            new_token_mask[:-1] = tok_in[1:]

        return {
            "input_ids": input_ids,
            "t_input_ids": t_input_ids,
            "span_offsets": self.span_offsets[index],
            "old_span_offsets": old_spans[index] if old_spans is not None else [],
            "old_token_mask": old_token_mask,
            "new_token_mask": new_token_mask,
            "offset_mapping": self.offset_mapping[index],
            "is_replay": is_replay
        }

    def _process_lm(self, i, samp, model_data, no_model_data, gen_data):
        input_ids = samp["input_ids"]
        source_len = 1
        
        prompt = None
        if self.args.model_type in ["qwen"] and 4294967295 in input_ids:
            source_len = np.where(input_ids==4294967295)[0][0]
            prompt = input_ids[:source_len]
            input_ids = np.concatenate([input_ids[:source_len], input_ids[source_len+1:]], axis=0)
        elif 65535 in input_ids:
            source_len = np.where(input_ids==65535)[0][0]
            prompt = input_ids[:source_len]
            input_ids = np.concatenate([input_ids[:source_len], input_ids[source_len+1:]], axis=0)
        
        input_ids = input_ids[:self.max_length]
        input_len = len(input_ids)
        model_data["input_ids"][i][:input_len-1] = torch.tensor(input_ids[:-1], dtype=torch.long)
        model_data["attention_mask"][i][:input_len-1] = 1.0
        if self.args.model_type in ["gpt2"]:
            model_data["position_ids"][i][:input_len-1] = torch.arange(0, input_len-1, dtype=torch.long)
        no_model_data["label"][i][:input_len-1] = torch.tensor(input_ids[1:], dtype=torch.long)
        no_model_data["label"][i][:source_len-1] = -100
        if "loss_mask" in no_model_data:
            no_model_data["loss_mask"][i][:input_len-1] = 1.0
            no_model_data["loss_mask"][i][:source_len-1] = 0
        
        if prompt is not None and gen_data is not None:
            gen_data["input_ids"][i][-len(prompt):] = torch.tensor(prompt, dtype=torch.long)
            gen_data["attention_mask"][i][-len(prompt):] = 1.0

    def move_to_device(self, model_data, no_model_data, gen_data, device):
        for k in model_data:
            model_data[k] = model_data[k].to(device)

        for k in no_model_data:
            if isinstance(no_model_data[k], torch.Tensor):
                no_model_data[k] = no_model_data[k].to(device)

        if gen_data is not None:
            for k in gen_data:
                gen_data[k] = gen_data[k].to(device)

        return model_data, no_model_data, gen_data

    def collate(self, samples):
        bs = len(samples)

        max_length = self.max_length
        
        model_data = {
            "input_ids": torch.ones(bs, max_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, max_length),
        }
        
        if self.args.model_type in ["gpt2"]:
            model_data["position_ids"] = torch.zeros(bs, max_length, dtype=torch.long)
            
        no_model_data = {
            "label": torch.ones(bs, max_length, dtype=torch.long) * -100,
            "loss_mask": torch.zeros(bs, max_length),
            "span_offsets": [sample["span_offsets"] for sample in samples],
            "old_span_offsets": [sample.get("old_span_offsets", []) for sample in samples],
            "old_token_mask": torch.stack([sample.get("old_token_mask", torch.zeros(max_length, dtype=torch.bool)) for sample in samples]),
            "new_token_mask": torch.stack([sample.get("new_token_mask", torch.zeros(max_length, dtype=torch.bool)) for sample in samples]),
            "offset_mapping": torch.concat([sample["offset_mapping"] for sample in samples]),
            "is_replay": torch.tensor([sample.get("is_replay", False) for sample in samples], dtype=torch.bool)
        }
        
        gen_data = {
            "input_ids": torch.ones(bs, self.max_prompt_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, self.max_prompt_length, dtype=torch.long),
        }

        for i, samp in enumerate(samples):
            self._process_lm(i, samp, model_data, no_model_data, gen_data)

        t_model_data, t_no_model_data = None, None
        if samples[0]["t_input_ids"] is not None:
            t_model_data = {
                "input_ids": torch.ones(bs, self.args.t_max_length, dtype=torch.long) * self.pad_id,
                "attention_mask": torch.zeros(bs, self.args.t_max_length),
            }
            
            if self.args.model_type in ["gpt2"]:
                t_model_data["position_ids"] = torch.zeros(bs, self.args.t_max_length, dtype=torch.long)
                
            t_no_model_data = {
                "label": torch.ones(bs, self.args.t_max_length, dtype=torch.long) * -100,
            }

            for i, samp in enumerate(samples):
                self._process_lm(i, {"input_ids": samp["t_input_ids"]}, t_model_data, t_no_model_data, None)

            # SD: teacher prompt tokens (the part before the sentinel), kept unpadded;
            # the sampled response is appended to them at train time.
            no_model_data["t_prompt_ids"] = [
                torch.tensor(samp["t_input_ids"][:np.where(samp["t_input_ids"] == 4294967295)[0][0]], dtype=torch.long)
                for samp in samples
            ]

        return model_data, no_model_data, gen_data, t_model_data, t_no_model_data


class LMEvalDataset(Dataset):
    def __init__(self, args, tokenizer, path, split, rng_sample: random.Random):
        self.args = args
        self.tokenizer = tokenizer
        self.split = split
        self.pad_id = self.tokenizer.eos_token_id
        self.max_length = args.max_length
        self.max_prompt_length = args.max_prompt_length
        self.rng_sample = rng_sample
        self.lm_ctx = DistributedMMapIndexedDataset(path, f"{split}", 0, 1)

        if os.path.exists(os.path.join(path, f"{split}.jsonl")):
            with open(os.path.join(path, f"{split}.jsonl")) as f:
                self.raw = [json.loads(line) for line in f.readlines()]
                self.answers = [x["response"] if isinstance(x["response"], list) else [x["response"]] for x in self.raw]
        
        self.num = len(self.lm_ctx)

        print(f"Num LM instances: {len(self.lm_ctx)}")

    def __len__(self):
        return self.num
   
    def __getitem__(self, index):
        return self._get_lm(index)
    
    def _get_lm(self, index):
        data = self.lm_ctx[index]
        input_ids = data.astype(int)
        return {
            "input_ids": input_ids
        }

    def _process_lm(self, i, samp, model_data, no_model_data, gen_data):
        input_ids = samp["input_ids"]
        source_len = 1
        
        prompt = None
        if self.args.model_type in ["qwen"] and 4294967295 in input_ids:
            source_len = np.where(input_ids==4294967295)[0][0]
            prompt = input_ids[:source_len]
            input_ids = np.concatenate([input_ids[:source_len], input_ids[source_len+1:]], axis=0)
        elif 65535 in input_ids:
            source_len = np.where(input_ids==65535)[0][0]
            prompt = input_ids[:source_len]
            input_ids = np.concatenate([input_ids[:source_len], input_ids[source_len+1:]], axis=0)
        
        input_ids = input_ids[:self.max_length]
        input_len = len(input_ids)
        model_data["input_ids"][i][:input_len-1] = torch.tensor(input_ids[:-1], dtype=torch.long)
        model_data["attention_mask"][i][:input_len-1] = 1.0
        if self.args.model_type in ["gpt2"]:
            model_data["position_ids"][i][:input_len-1] = torch.arange(0, input_len-1, dtype=torch.long)
        no_model_data["label"][i][:input_len-1] = torch.tensor(input_ids[1:], dtype=torch.long)
        no_model_data["label"][i][:source_len-1] = -100
        no_model_data["loss_mask"][i][:input_len-1] = 1.0
        no_model_data["loss_mask"][i][:source_len-1] = 0
        
        if prompt is not None:
            gen_data["input_ids"][i][-len(prompt):] = torch.tensor(prompt, dtype=torch.long)
            gen_data["attention_mask"][i][-len(prompt):] = 1.0

    def move_to_device(self, model_data, no_model_data, gen_data, device):
        for k in model_data:
            model_data[k] = model_data[k].to(device)

        for k in no_model_data:
            no_model_data[k] = no_model_data[k].to(device)

        for k in gen_data:
            gen_data[k] = gen_data[k].to(device)

        return model_data, no_model_data, gen_data

    def collate(self, samples):
        bs = len(samples)

        max_length = self.max_length
        
        model_data = {
            "input_ids": torch.ones(bs, max_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, max_length),
        }
        
        if self.args.model_type in ["gpt2"]:
            model_data["position_ids"] = torch.zeros(bs, max_length, dtype=torch.long)
            
        no_model_data = {
            "label": torch.ones(bs, max_length, dtype=torch.long) * -100,
            "loss_mask": torch.zeros(bs, max_length)
        }
        
        gen_data = {
            "input_ids": torch.ones(bs, self.max_prompt_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, self.max_prompt_length, dtype=torch.long),
        }

        for i, samp in enumerate(samples):
            self._process_lm(i, samp, model_data, no_model_data, gen_data)
        
        return model_data, no_model_data, gen_data
