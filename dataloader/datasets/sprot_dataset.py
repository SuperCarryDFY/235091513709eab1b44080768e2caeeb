import json
import random

import numpy as np
import lmdb
import torch.distributed as dist
import csv
import os
from tqdm import tqdm 
from dataloader.datasets.base_dataset import BaseDataset
from utils.constants import sequence_level, sequence_level2
from utils.dataloader_utils import record2text
from utils.path_utils import env_path

# 不放回加权采样
def a_expj_sample(prob, m):
    """ 根据 prob 数组无放回随机抽取 m 个元素 """
    import heapq
    import math
    topn = []
    for i, pi in enumerate(prob[:m]):
        heapq.heappush(topn, (random.random() ** (1/pi), i))
        
    thres, w_sum = topn[0][0], 0
    xw = math.log(random.random()) / math.log(thres)
    i = m
    for pi in prob[m:]:
        if w_sum + pi >= xw:
            tw = thres ** pi
            r2 = random.uniform(tw, 1)
            ki = r2 ** (1/pi)
            heapq.heappop(topn)
            heapq.heappush(topn, (ki, i))
            thres = topn[0][0]
            xw = math.log(random.random()) / math.log(thres)
            w_sum = 0
        else:
            w_sum += pi
        i += 1
    return [item[1] for item in topn]


class SProtDataset(BaseDataset):
    def __init__(
        self,
        split,
        sprot_data_path,
        sprot_dataset_dir,
        sprot_text_data_path,
        template_path,
        max_aa_seq_len,
        max_text_seq_len,
        splits_sub_dir='splits',
        paragraph2sentence_path=None,
        paraphrased_texts_path=None,
        protein_level_only=False,
        return_records=False,
        seq_type="protein_sequence",
        use_cluster_weight=False,
        length_info_rate=0,
        random_template_path=None,
        _log=None
    ):
        base_data_kwargs = {"length_info_rate": length_info_rate,
                            "random_template_path": random_template_path}
        super().__init__(base_data_kwargs)
        self.max_aa_seq_len = max_aa_seq_len
        self.max_text_seq_len = max_text_seq_len
        self.indices = (
            open(f"{sprot_dataset_dir}/{splits_sub_dir}/{split}.tsv", "r")
            .read()
            .strip()
            .split("\n")
        )
        
        # read two paragraph2sentence files
        self.p2s_dict = json.load(open(paragraph2sentence_path, "r"))
        self.paraphrased_texts = json.load(open(paraphrased_texts_path, "r"))
        
        seq_env = lmdb.open(sprot_data_path, lock=False, map_size=1024**4)
        self.id2seq = seq_env.begin()  # get a operator
        
        # read text file
        text_env = lmdb.open(sprot_text_data_path, lock=False, map_size=1024**4)
        self.id2text = text_env.begin()

        # load cluster weight
        self.use_cluster_weight = use_cluster_weight
        if use_cluster_weight:
            weight_path = f"{sprot_dataset_dir}/{split}_weights.tsv"
            assert os.path.exists(weight_path), f"There is not this file {weight_path} when specficing use_cluster_weight=True"
            self.cluster_weight = []
            with open(weight_path, 'r') as tsvfile:
                reader = csv.reader(tsvfile, delimiter='\t')
                # next(reader)  # Skip header if there is one
                for row in tqdm(reader, total=len(self.indices), desc="reading SwissProt cluster weights file", disable=dist.get_rank() != 0):
                    weight = float(row[1])
                    self.cluster_weight.append(weight)

            if not np.isclose(np.sum(self.cluster_weight), 1):
                if _log:
                    _log.info("The sum of the cluster weight should be 1, reweighting...")
                sum_weight = np.sum(self.cluster_weight)
                self.cluster_weight = [i / sum_weight for i in self.cluster_weight]        
        # read template
        with open(template_path, "r") as f:
            self.template = json.load(f)
        
        self.seq_type = seq_type
        self.protein_level_only = protein_level_only
        self.return_records = return_records
        self.index_mapper = self.indices

        if _log:
            _log.info(f"SwissProt {split} ids: {len(self.indices)}")


    def __getitem__(self, index):
        suite = self.get_suite(index)
        return suite

    def get_suite(self, index):
        ret = {"row_index": index, "uni_id": self.indices[index]}
        ret.update(self.get_seq(ret["uni_id"]))
        ret.update(self.get_text(ret["uni_id"]))
        return ret

    def get_text(self, uni_id):
        try:
            all_texts = eval(self.id2text.get(uni_id.encode()).decode())["text"].split("\n")[:-1]
        except:
            print(uni_id)
            assert 0
        if self.protein_level_only:
            protein_level_all_texts = []
            for text in all_texts:
                subsection = text.split("\t")[3]
                if subsection in sequence_level:
                    protein_level_all_texts.append(text)
            all_texts = protein_level_all_texts
        
        if random.random() < 0.5:
            sample_k = min(len(all_texts), 10)
        else:
            sample_k = 1
        texts = random.sample(all_texts, k=sample_k)

        ripe_text = []
        subsections = []
        if self.return_records:
            records = []
        for text in texts:
            row = text.split("\t")
            cur_text, subsection, record = self.get_text_from_one_row(row)
            ripe_text.append(cur_text)
            subsections.append(subsection)
            if self.return_records:
                records.append(record)
        res = {
            "text": " ".join(ripe_text),
            # "subsections": subsections,
        }
        if self.return_records:
            res["records"] = records
        return res

    def get_text_from_one_row(self, row):
        if len(row) == 1:
            return row[0], None, None
        subsection, raw_text, notes = row[3], row[5], row[6]
        record = [subsection, raw_text, notes]
        cur_text = record2text(record, self.template)
        if type(cur_text) == list:
            cur_text = random.choice(cur_text)

        # If we haven't rewrite the text accourding to the template:
        if cur_text == raw_text and subsection in self.p2s_dict:
            sentences = self.p2s_dict[subsection][cur_text]
            paraphrased_text_list = []
            for s in sentences:
                pool = self.paraphrased_texts[subsection][s] + [s]
                paraphrased_text_list.append(np.random.choice(pool))

            cur_text = " ".join(paraphrased_text_list)
        if not cur_text.endswith("."):
            cur_text += "."
        return cur_text, subsection, record

    def check_length(self, seq):
        if len(seq) > self.max_aa_seq_len:
            start = random.randint(0, len(seq) - self.max_aa_seq_len)
            end = start + self.max_aa_seq_len
            seq = seq[start:end]
        else:
            start = 0
            end = len(seq)
        return seq, start, end

    def get_seq(self, uni_id):
        if self.seq_type == "protein_sequence":
            return self.get_protein_seq(uni_id)
        elif self.seq_type == "structure_token":
            return self.get_structure_token(uni_id)
        elif self.seq_type == "structure_aware_token":
            return self.get_structure_aware_token(uni_id)
        else:
            raise ValueError(f"Unknown seq_type: {self.seq_type}")

    def get_protein_seq(self, uni_id):
        seq = eval(self.id2seq.get(uni_id.encode()).decode())['seq']
        seq, start, end = self.check_length(seq)
        
        return {
            "uni_id": uni_id,
            "prot": seq,
            "start": start,
            "end": end,
        }
    
    def get_structure_token(self, uni_id):
        seq = eval(self.id2seq.get(uni_id.encode()).decode())['foldseek_seq']
        seq, start, end = self.check_length(seq)
        
        return {
            "uni_id": uni_id,
            "structure_token": seq,
            "start": start,
            "end": end,
        }
    
    def get_structure_aware_token(self, uni_id):
        seq = eval(self.id2seq.get(uni_id.encode()).decode())['seq']
        structure_token = eval(self.id2seq.get(uni_id.encode()).decode())['foldseek_seq']
        seq, start, end = self.check_length(seq)
        structure_token = structure_token[start:end]
        
        res = {
            "uni_id": uni_id,
            "prot": "".join(a + b for a, b in zip(seq, structure_token)),
            "start": start,
            "end": end,
        }
        return res

        

    def get_weights(self, weight=1):
        # make sure the sum of the weights is 1.
        if self.use_cluster_weight:
            return [weight* w for w in self.cluster_weight]
        else:
            return [weight / len(self) for _ in range(len(self))]


if __name__ == "__main__":
    ds = SProtDataset(
        split="train",
        sprot_data_path=env_path("DATA_ROOT", "sprot_text/LMDB/swissprot_group50"),
        sprot_dataset_dir=env_path("DATA_ROOT", "sprot_text"),
        sprot_text_data_path=env_path("DATA_ROOT", "sprot_text/LMDB/meta_data_20231129"),
        template_path=env_path("DATA_ROOT", "sprot_text/train_text_template.json"),
        max_aa_seq_len=1024,
        max_text_seq_len=1024,
        paragraph2sentence_path=env_path("DATA_ROOT", "sprot_text/paragraph2sentence.json"),
        paraphrased_texts_path=env_path("DATA_ROOT", "sprot_text/train_paraphrased_texts.json"),
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    for item in ds:
        print(item)
        break
