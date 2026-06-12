import torch
from tokenizers import Tokenizer
from transformers import AutoTokenizer, EsmTokenizer, T5Tokenizer

from utils.dataloader_utils import get_pretrained_tokenizer
from utils.path_utils import env_path
from transformers.utils import logging

from dataloader.datasets import SProtDataset

try:
    from .datamodule_base import BaseDataModule
except:
    from datamodule_base import BaseDataModule

logger = logging.get_logger(__name__)


class SProtModule(BaseDataModule):
    def __init__(self, model_cfg, datasets_cfg, training_cfg, istraining=True, _log=None):
        super().__init__(training_cfg)
        self.max_aa_seq_len = model_cfg["max_aa_seq_len"]
        self.max_text_seq_len = model_cfg["max_text_seq_len"]
        self.sprot_data_path = datasets_cfg["sprot_data_path"]
        self.sprot_dataset_dir = datasets_cfg["sprot_dataset_dir"]
        self.sprot_text_data_path = datasets_cfg["sprot_text_data_path"]
        self.train_template_path = datasets_cfg["train_template_path"]
        self.test_template_path = datasets_cfg["test_template_path"]
        self.paragraph2sentence_path = datasets_cfg["paragraph2sentence_path"]
        self.train_paraphrased_texts_path = datasets_cfg["train_paraphrased_texts_path"]
        self.test_paraphrased_texts_path = datasets_cfg["test_paraphrased_texts_path"]
        self.seq_type = datasets_cfg.get("seq_type", "protein_sequence")
        self.splits_sub_dir=datasets_cfg["splits_sub_dir"]
        self.protein_level_only = datasets_cfg.get("protein_level_only", False)
        self.return_records = datasets_cfg.get("return_records", False)
        # length info rate
        self.length_info_rate = datasets_cfg.get("length_info_rate", 0)
        self.random_template_path = datasets_cfg.get("random_template_path", None)
        # afdb plddt consideration 
        self.mask_text_ratio = datasets_cfg.get("mask_text_ratio", 0)
        self._log = _log
        self.training = istraining

        self.has_hard_val = datasets_cfg.get("hard_val_metrics", False)
        
        self.tokenizer = EsmTokenizer.from_pretrained(model_cfg["tokenizer"])

        # load lm tokenizer
        self.text_tokenizer = get_pretrained_tokenizer(model_cfg["lm"], model_type=model_cfg["lm_type"])

    def setup_train_dataset(self):
        assert self.training
        self.set_train_dataset()
        self.train_dataset.tokenizer = self.tokenizer
        self.train_dataset.text_tokenizer = self.text_tokenizer
        self.setup_flag = True

    def setup(self):
        if not self.setup_flag:
            if self.training:
                self.set_train_dataset()
                self.set_val_dataset()
                self.train_dataset.tokenizer = self.tokenizer
                self.train_dataset.text_tokenizer = self.text_tokenizer
                self.val_dataset.tokenizer = self.tokenizer
                self.val_dataset.text_tokenizer = self.text_tokenizer
                if self.has_hard_val:
                    self.set_hard_val_dataset()
                    self.hard_val_dataset.tokenizer = self.tokenizer
                    self.hard_val_dataset.text_tokenizer = self.text_tokenizer

            else:
                self.set_test_dataset()
                self.test_dataset.tokenizer = self.tokenizer
                self.test_dataset.text_tokenizer = self.text_tokenizer

            self.setup_flag = True

    @property
    def dataset(self):
        return SProtDataset

    @property
    def dataset_name(self):
        return "SwissProtDatasetWithText"

    def set_train_dataset(self):
        self.train_dataset = self.dataset(
            split="train",
            sprot_data_path=self.sprot_data_path,
            sprot_dataset_dir=self.sprot_dataset_dir,
            sprot_text_data_path=self.sprot_text_data_path,
            template_path=self.train_template_path,
            max_aa_seq_len=self.max_aa_seq_len,
            max_text_seq_len=self.max_text_seq_len,
            splits_sub_dir=self.splits_sub_dir,
            paragraph2sentence_path=self.paragraph2sentence_path,
            paraphrased_texts_path=self.train_paraphrased_texts_path,
            protein_level_only=self.protein_level_only,
            seq_type=self.seq_type,
            return_records=self.return_records,
            length_info_rate = self.length_info_rate,
            random_template_path = self.random_template_path,
            _log=self._log
        )

    def set_val_dataset(self):
        self.val_dataset = self.dataset(
            split="val",
            sprot_data_path=self.sprot_data_path,
            sprot_dataset_dir=self.sprot_dataset_dir,
            sprot_text_data_path=self.sprot_text_data_path,
            template_path=self.train_template_path,
            max_aa_seq_len=self.max_aa_seq_len,
            max_text_seq_len=self.max_text_seq_len,
            splits_sub_dir=self.splits_sub_dir,
            paragraph2sentence_path=self.paragraph2sentence_path,
            paraphrased_texts_path=self.train_paraphrased_texts_path,
            protein_level_only=self.protein_level_only,
            seq_type=self.seq_type,
            return_records=self.return_records,
            _log=self._log
        )

    def set_test_dataset(self):
        self.test_dataset = self.dataset(
            split="test",
            sprot_data_path=self.sprot_data_path,
            sprot_dataset_dir=self.sprot_dataset_dir,
            sprot_text_data_path=self.sprot_text_data_path,
            template_path=self.test_template_path,
            max_aa_seq_len=self.max_aa_seq_len,
            max_text_seq_len=self.max_text_seq_len,
            splits_sub_dir=self.splits_sub_dir,
            paragraph2sentence_path=self.paragraph2sentence_path,
            paraphrased_texts_path=self.test_paraphrased_texts_path,
            protein_level_only=self.protein_level_only,
            seq_type=self.seq_type,
            return_records=self.return_records,
            _log=self._log
        )


    def set_hard_val_dataset(self):
        self.hard_val_dataset = self.dataset(
            split="hard_val",
            sprot_data_path=self.sprot_data_path,
            sprot_dataset_dir=self.sprot_dataset_dir,
            sprot_text_data_path=self.sprot_text_data_path,
            template_path=self.test_template_path,
            max_aa_seq_len=self.max_aa_seq_len,
            max_text_seq_len=self.max_text_seq_len,
            splits_sub_dir=self.splits_sub_dir,
            paragraph2sentence_path=self.paragraph2sentence_path,
            paraphrased_texts_path=self.test_paraphrased_texts_path,
            protein_level_only=self.protein_level_only,
            seq_type=self.seq_type,
            return_records=self.return_records,
            _log=self._log
        )

    def set_hard_test_dataset(self):
        self.hard_test_dataset = self.dataset(
            split="hard_test",
            sprot_data_path=self.sprot_data_path,
            sprot_dataset_dir=self.sprot_dataset_dir,
            sprot_text_data_path=self.sprot_text_data_path,
            template_path=self.test_template_path,
            max_aa_seq_len=self.max_aa_seq_len,
            max_text_seq_len=self.max_text_seq_len,
            splits_sub_dir=self.splits_sub_dir,
            paragraph2sentence_path=self.paragraph2sentence_path,
            paraphrased_texts_path=self.test_paraphrased_texts_path,
            protein_level_only=self.protein_level_only,
            seq_type=self.seq_type,
            return_records=self.return_records,
            _log=self._log
        )
