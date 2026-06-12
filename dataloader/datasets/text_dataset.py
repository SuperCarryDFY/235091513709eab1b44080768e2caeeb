import json
import os
import random

import pandas as pd
import torch
import torch.distributed as dist
from Bio import SeqIO
from tqdm import tqdm
from Bio import SeqIO
from utils.dataloader_utils import record2text
import lmdb
from torch.utils.data import Dataset
from .base_dataset import BaseDataset



class TextDataset(Dataset):
    def __init__(self, data_path, sprot_data_path, text_tokenizer, text_max_sequence_len, skip_batches=0):
        self.data_path = data_path

        lines = open(self.data_path, "r").readlines()
        text_sequence_list = []
        # Read Text File
        for line in lines[1 + skip_batches:]: # remove headline
            text_sequence = line.strip().split("\t")[-1]
            UniID = line.strip().split("\t")[0]
            text_sequence_list.append((UniID, text_sequence))
        # Read ID2GT File
        seq_env = lmdb.open(sprot_data_path, lock=False, map_size=1024**4)
        self.id2seq = seq_env.begin()  # get a operator
        self.text_sequence_list = text_sequence_list
        self.text_tokenizer = text_tokenizer
        self.text_max_sequence_len = text_max_sequence_len

    def __getitem__(self, index):
        UniID = self.text_sequence_list[index][0]
        # Just in case QA format.
        text_sequence = self.text_sequence_list[index][1].replace("<QA/n>", "\n")
        text_sequence_encode = self.text_tokenizer(
            text_sequence,
            truncation=True,
            max_length=self.text_max_sequence_len,
            padding="max_length",
            return_tensors="pt",
        )
        text_sequence_input_ids = text_sequence_encode.input_ids.squeeze()
        text_sequence_attention_mask = text_sequence_encode.attention_mask.squeeze()
        
        gt_str = self.id2seq.get(UniID.encode())
        if gt_str:
            gt = eval(gt_str.decode()).get("foldseek_seq", "None")
            gt_seq = eval(gt_str.decode()).get("seq", "None")
        else:
            gt = "None"
            gt_seq = "None"

        batch = {
            "text": text_sequence,
            "text_ids": text_sequence_input_ids,
            "text_masks": text_sequence_attention_mask,
            "gt": gt,
            "gt_seq": gt_seq
        }

        return batch

    def __len__(self):
        return len(self.text_sequence_list)
    #  We need text_ids, text_masks, text, prot in a batch



class SProtTextDataset(BaseDataset):
    def __init__(
        self,
        split,
        sprot_dataset_dir,
        sprot_text_data_path,
        template_path,
        max_text_seq_len,
        splits_sub_dir='splits',
        tasks={"mlm": 1},
    ):
        super().__init__()
        assert split in ["train", "val", "test", "test_sample"]
        self.split = split
        self.max_text_seq_len = max_text_seq_len


        self.indices = (
            open(f"{sprot_dataset_dir}/{splits_sub_dir}/{split}.tsv", "r")
            .read()
            .strip()
            .split("\n")
        )
        # read text file
        self.text_table = pd.read_csv(
            sprot_text_data_path, sep="\t", header=None, keep_default_na=False
        )
        self.text_table = self.text_table[self.text_table[0].isin(self.indices)]
        self.text_table = self.text_table.reset_index()
        # read template
        with open(template_path, "r") as f:
            self.template = json.load(f)
        self.index_mapper = self.text_table
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"{split} sprot ids:", len(self.indices))
            print(f"{split} all text:", len(self.text_table))

        self.tasks = tasks

    def __getitem__(self, index):
        suite = self.get_suite(index)
        return suite

    def get_suite(self, index):

        ret = dict()
        ret.update(self.get_text(index))
        return ret

    def get_text(self, raw_index):
        row = self.text_table.iloc[raw_index, :]
        uni_id = row[0]
        record = [row[3], row[5], row[6]]
        ripe_text = record2text(record, self.template)
        if type(ripe_text) == list:
            ripe_text = random.choice(ripe_text)
        return {
            "text": ripe_text,
            "uni_id": uni_id,
            "raw_index": raw_index,
        }
