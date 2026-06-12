import torch
import lmdb
import numpy as np
import json
import random

mask_strategy_dict = {
    "woAA-partialstructure": 0,
    "woAA-fullstructure":  1,
    "partialAA-wostructure": 2,
    "partialAA-partialstructure": 3,
    "partialAA-fullstructure": 4,
}

class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        kwargs=None,
    ):
        super().__init__()
        if kwargs is not None:
            self.mask_text_ratio = kwargs.get("mask_text_ratio", 0)
            self.length_info_rate = kwargs.get("length_info_rate", 0)
            self.plddt_threshold = kwargs.get("plddt_threshold", 0)
            if self.length_info_rate > 0:
                with open(kwargs["random_template_path"], 'r') as file:
                    json_data = json.load(file)
                    self.length_sentences = json_data["sequence_length"]
                #   selected_text = random.choice(self.length_sentences).format(seq_len=seq_len)
        
    def __len__(self):
        return len(self.index_mapper)
    
    def get_plddt_mask(self, uni_id):
        try:
            plddt_list = np.array(eval("["  + eval(self.txn_afdb_plddt.get(uni_id.encode()).decode())["plddt"] + "]"))
            mask = np.where(plddt_list >= self.plddt_threshold, 1, 0)
        except:
            mask = None
        return mask
    
    def plddt_mask(self, structure_token_seq, uni_id, start, end):
        mask = self.get_plddt_mask(uni_id)
        if mask is None:
            return structure_token_seq
        else:
            mask = mask[start:end]
        assert len(mask) == len(structure_token_seq), f"{uni_id}: {len(mask)} != {len(structure_token_seq)}, {start}, {end}"
        return "".join([structure_token_seq[i] if mask[i] == 1 else "#" for i in range(len(mask))])

    def plddt_mask4mask_prot(self, mask_prot_seq, uni_id, start, end):
        mask = self.get_plddt_mask(uni_id)
        if mask is None:
            return mask_prot_seq
        else:
            mask = mask[start:end]
        structure_token_seq = mask_prot_seq.replace("#", "")
        assert len(mask) == len(structure_token_seq), f"{uni_id}: {len(mask)} != {len(structure_token_seq)}, {start}, {end}"
        mask_structure_token_seq = "".join([structure_token_seq[i] if mask[i] == 1 else "#" for i in range(len(mask))])
        return "#" + "#".join(list(mask_structure_token_seq))

    def add_length_info(self, dict_batch):
        aug_texts = []
        texts = dict_batch["text"]
        for idx, text in enumerate(texts):
            seq_len = len(dict_batch["structure_token"][idx])
            seq_len = round(seq_len / 100) * 100
            sequence_length_sentence = random.choice(self.length_sentences).format(seq_len=seq_len)[:-1]
            text_list = text.split(".")
            text_list = [t for t in text_list if len(t) > 0]
            text_list.append(sequence_length_sentence)
            random.shuffle(text_list)
            selected_text = ".".join(text_list) + "."
            aug_texts.append(selected_text)
        dict_batch["text"] = aug_texts
        return aug_texts

    def collate(self, batch):
        keys = [key for key in batch[0].keys()]
        # dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}
        dict_batch = {}
        # Here we modify structure token by plddt mask 
        for k in keys:
            dict_batch[k] = [dic[k] if k in dic else None for dic in batch]
        # add seq len information to text
        if random.random() < self.length_info_rate:
            self.add_length_info(dict_batch)
        
        # encode text
        # randomly replace text in dict_batch with ""
        dict_batch["text"] = [i if random.random() > self.mask_text_ratio else "" for i in dict_batch["text"]]
        encodings = self.text_tokenizer(
            dict_batch["text"],
            padding="longest", # max_length,longest
            truncation=True,
            max_length=self.max_text_seq_len,
            return_tensors="pt",
        )
        dict_batch[f"text_ids"] = encodings["input_ids"]
        dict_batch[f"text_masks"] = encodings["attention_mask"]

        # encode protein
        key = "prot" if "prot" in keys else "structure_token"
        encodings = self.tokenizer(
            dict_batch[key],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_aa_seq_len,
            padding="longest", # max_length,longest
        )
        dict_batch.update({
            f"{key}_ids": encodings.input_ids,
            f"{key}_masks": encodings.attention_mask,
        })
        # encode mask prot ids if possible

        if "mask_prot" in keys:
            encodings = self.tokenizer(
                dict_batch["mask_prot"],
                return_tensors="pt",
                truncation=True,
                max_length=self.max_aa_seq_len,
                padding="longest", # max_length,longest
            )
            dict_batch.update({
                "mask_prot_ids": encodings.input_ids,
                "mask_prot_masks": encodings.attention_mask,
            })
        # replace mask_strategy from str to tensor
        if "mask_strategy" in keys:
            dict_batch["mask_strategy"] = torch.tensor([mask_strategy_dict[i] for i in dict_batch["mask_strategy"]])    
        return dict_batch
