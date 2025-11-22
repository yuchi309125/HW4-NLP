import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")

        # paths
        nl_path = os.path.join(data_folder, f"{split}.nl")
        self.nl_lines = load_lines(nl_path)

        if split != "test":
            sql_path = os.path.join(data_folder, f"{split}.sql")
            self.sql_lines = load_lines(sql_path)
        else:
            self.sql_lines = None

        # process
        self.data = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        data = []
        bos_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")

        for i, nl in enumerate(self.nl_lines):
            enc = tokenizer.encode(nl, add_special_tokens=True)

            if split != "test":
                sql = self.sql_lines[i]
                dec = tokenizer.encode(sql, add_special_tokens=True)
                # Shift decoder input: prepend BOS
                dec_input = [bos_id] + dec
                target = dec + [tokenizer.eos_token_id]
            else:
                dec_input = None
                target = None

            data.append({
                "encoder_ids": torch.tensor(enc),
                "decoder_inputs": torch.tensor(dec_input) if dec_input else None,
                "decoder_targets": torch.tensor(target) if target else None
            })

        return data

    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def normal_collate_fn(batch):
    encoder_ids = [item["encoder_ids"] for item in batch]
    decoder_inputs = [item["decoder_inputs"] for item in batch]
    decoder_targets = [item["decoder_targets"] for item in batch]

    encoder_ids = pad_sequence(encoder_ids, batch_first=True, padding_value=0)
    decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value=0)
    decoder_targets = pad_sequence(decoder_targets, batch_first=True, padding_value=0)

    encoder_mask = (encoder_ids != 0).long()

    # first token for evaluation (usually <extra_id_0>)
    initial_decoder_inputs = decoder_inputs[:, 0].unsqueeze(1)

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs


def test_collate_fn(batch):
    encoder_ids = [item["encoder_ids"] for item in batch]
    encoder_ids = pad_sequence(encoder_ids, batch_first=True, padding_value=0)
    encoder_mask = (encoder_ids != 0).long()

    bos_id = 0  # <extra_id_0>, tokenizer will map this
    initial_decoder_inputs = torch.tensor([[bos_id]] * len(batch))

    return encoder_ids, encoder_mask, initial_decoder_inputs


def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    return train_x, train_y, dev_x, dev_y, test_x