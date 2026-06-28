import sys
import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from typing import List, Dict, Tuple
from transformers import DataCollatorForSeq2Seq

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config_gen import GenTrainingConfig
import pandas as pd

FIXED_DATA_PATH = r'./pre data/'
name_key_match = {
    'sbzjs': '上班这件事',
    'rswtyjs': '人生问题研究社',
    'rjqlgc': '人间情侣观察',
    'zctytlxz': '职场体验讨论小组',
    'zctcdh': '职场吐槽大会',
}

def build_input_c2r(data_name: str) -> Tuple[List[Tuple], Dict]:
    """Load raw CSVs and build the unified list of tuples and block map.

    Returns:
        all_data: List of tuples (postID, post_content, comment, review, label)
        block_map: Dict mapping postID to list of indices in all_data
    """
    value, key = name_key_match[data_name], data_name

    path_pkl = rf"{FIXED_DATA_PATH}{value}/{data_name}_cache.pkl"
    with open(path_pkl, "rb") as f:
        cache_data = pickle.load(f)
    return cache_data

# ─── Unified Dataset ──────────────────────────────────────────────────────────

class GenerationDataset(Dataset):
    """Unified dataset for both training and evaluation.

    Accepts the list of raw tuples produced by build_input_c2r and
    tokenises each sample at construction time into:
        context_input_ids  — prompt only (for .generate())
        full_input_ids     — prompt + response (for PPL / Trainer)
        full_labels        — -100 on prompt tokens, token ids on response
        reference          — gold comment string
        history            — post body string
        sample_idx         — postID
    """

    SYSTEM_CONTENT = (
        "你是一个在职场论坛里冲浪的真实网友。请根据楼主的帖子内容，"
        "请根据帖子的内容和生成回复。"
    )

    def __init__(self, data: List[Tuple], tokenizer, max_seq_len: int = 1024):
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len
        self.all_data    = data   # raw tuples — tokenisation is lazy in __getitem__

        assert tokenizer.pad_token_id is not None, \
            "tokenizer.pad_token_id must be set before building the dataset."
        print(f"Max seq length be set to {self.max_seq_len} for dataset tokenisation.")

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        pid, post_content, comment, review, label = self.all_data[idx]

        ctx  = self.SYSTEM_CONTENT
        full_info = ctx + f"\n帖子内容:{post_content}"

        context_messages = [{'role': 'user', 'content': full_info}]
        full_messages = [
            {'role': 'user',      'content': full_info},
            {'role': 'assistant', 'content': comment},
        ]

        context_ids = self.tokenizer.apply_chat_template(
            context_messages, tokenize=True, add_generation_prompt=True,
            max_length=self.max_seq_len, truncation=True,
        )
        full_ids = self.tokenizer.apply_chat_template(
            full_messages, tokenize=True, add_generation_prompt=False,
            max_length=self.max_seq_len, truncation=True,
        )

        # if len(full_ids) > self.max_seq_len:
        #     full_ids = full_ids[:self.max_seq_len]

        prompt_len = len(context_ids)
        if prompt_len >= len(full_ids):          # edge case: clamp so ≥1 response token has a real label
            prompt_len = max(0, len(full_ids) - 1)

        # print(f"Tokenised sample {idx}: prompt_len={prompt_len}, full_len={len(full_ids)}")

        labels = [-100] * prompt_len + full_ids[prompt_len:]

        return {
            "sample_idx":        pid,
            "context_input_ids": torch.tensor(context_ids, dtype=torch.long),
            "full_input_ids":    torch.tensor(full_ids,    dtype=torch.long),
            "full_labels":       torch.tensor(labels,      dtype=torch.long),
            "reference":         comment,
            "postkey":           post_content,
        }


# ─── Unified Collator ─────────────────────────────────────────────────────────

class GenerationCollator:
    """Unified collator for both training (HF Trainer) and evaluation.

    Training keys (standard HF Trainer names):
        input_ids        — full sequence, right-padded
        attention_mask   — full sequence mask
        labels           — -100 on prompt, token ids on response (right-padded)

    Evaluation / generation keys (aliases of the above):
        context_input_ids      — prompt only, left-padded (for .generate())
        context_attention_mask
        full_input_ids         — same tensor as input_ids
        full_attention_mask    — same tensor as attention_mask
        full_labels            — same tensor as labels

    Pass-through (lists):
        reference, history, sample_idx
    """

    def __init__(self, tokenizer, **kwargs):
        self.tokenizer     = tokenizer
        self.base_collator = DataCollatorForSeq2Seq(
            tokenizer, label_pad_token_id=-100, **kwargs
        )

    def __call__(self, features):
        # Left-pad context for .generate()
        original_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = 'left'
        context_batch = self.tokenizer.pad(
            [{"input_ids": f["context_input_ids"]} for f in features],
            padding=True,
            return_tensors="pt",
        )
        self.tokenizer.padding_side = original_side

        # Right-pad full sequence + labels for training / PPL
        full_batch = self.base_collator(
            [{"input_ids": f["full_input_ids"], "labels": f["full_labels"]} for f in features]
        )

        return {
            # HF Trainer standard keys
            "input_ids":              full_batch["input_ids"],
            "attention_mask":         full_batch["attention_mask"],
            "labels":                 full_batch["labels"],
            # Explicit eval aliases
            "context_input_ids":      context_batch["input_ids"],
            "context_attention_mask": context_batch["attention_mask"],
            # Metadata
            "reference":  [f["reference"]  for f in features],
            "postkey":    [f["postkey"]    for f in features],
            "sample_idx": [f["sample_idx"] for f in features],
        }


# ─── Loader factory ───────────────────────────────────────────────────────────

def gen_loader_warp(data_name: str, tokenizer, config: GenTrainingConfig):
    """Build train / val / test DataLoaders for a forum dataset.

    Args:
        data_name: key in name_key_match, e.g. 'sbzjs'
        tokenizer: HF tokenizer
        config:    GenTrainingConfig

    Returns:
        train_loader, val_loader, test_loader, (train_ds, val_ds, test_ds)
    """
    # 1. Load all raw tuples once
    all_data, block_map = build_input_c2r(data_name)
    all_block_idx = np.array(list(block_map.keys()))

    # 2. Train / Val / Test split
    num_total = len(all_block_idx)
    train_blocks: list = []
    val_blocks:   list = []
    test_blocks:  list = []


    n_train = int(0.8 * num_total)
    n_val   = int(0.1 * num_total)
    
    train_blocks = list(all_block_idx[:n_train])
    val_blocks   = list(all_block_idx[n_train: n_train + n_val])
    test_blocks  = list(all_block_idx[n_train + n_val:])
    
    if config.fast_train:
        n_train = int(0.1 * num_total)
        train_blocks = list(all_block_idx[:n_train])

    if config.few_shot or config.semi_supervised:
        # FSL / SSL: val and test are fixed fractions of total; rest is train
        print(f"Few-shot / Semi-supervised mode: using fixed val/test ratios, rest train.")
        r_val  = getattr(config, 'val_ratio',  0.1)
        r_test = getattr(config, 'test_ratio', 0.1)
        r_train = getattr(config, 'semi_ratio', 0.1)
        n_val  = int(num_total * r_val)
        n_test = int(num_total * r_test)

        shuffled = np.random.permutation(all_block_idx)
        val_blocks   = list(shuffled[:n_val])
        test_blocks  = list(shuffled[n_val: n_val + n_test])
        train_blocks = list(shuffled[n_val + n_test: n_val + n_test + int(num_total * r_train)])


    print(f"Gen Data Split for post: Train {len(train_blocks)}, Val {len(val_blocks)}, Test {len(test_blocks)} blocks. total {num_total} blocks.")

    def _make_dataset(blocks):
        subset = []
        for pid in blocks:
            for idx in block_map[pid]:
                subset.append(all_data[idx])
        return GenerationDataset(subset, tokenizer, config.max_seq_length)

    # 4. Build datasets
    train_ds = _make_dataset(train_blocks)
    val_ds   = _make_dataset(val_blocks)
    test_ds  = _make_dataset(test_blocks)

    collator = GenerationCollator(tokenizer)

    # 5. Build loaders
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, collate_fn=collator)
    val_loader   = DataLoader(val_ds,   batch_size=config.batch_size, shuffle=False,
                              num_workers=config.num_workers, collate_fn=collator)
    test_loader  = DataLoader(test_ds,  batch_size=config.batch_size, shuffle=False,
                              num_workers=config.num_workers, collate_fn=collator)

    return train_loader, val_loader, test_loader, (train_ds, val_ds, test_ds)

