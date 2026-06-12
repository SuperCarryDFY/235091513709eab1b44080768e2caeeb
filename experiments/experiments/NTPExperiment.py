from experiments import BaseExperiment
from utils import experiments_utils as eu
from utils import objectives
from transformers import EsmTokenizer, AutoTokenizer
from functools import partial
import torch 
import os
from tqdm import tqdm
import torch.distributed as dist
from utils import sample
from accelerate.utils import gather_object
from utils.path_utils import hf_cache_dir, resolve_pretrained_path

OBJECTIVES = {"next_token_prediction": objectives.compute_next_token_prediction,
              "next_structure_token_prediction": objectives.compute_next_structure_token_prediction}

class NTPExperiment(BaseExperiment):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        
    def _sample_dataloader(self) -> None:
        return eu.create_sample_dataloader(self.cfg.dataset, self.cfg.sample, self.text_tokenizer)


    def _optimizer(self):
        return eu.configure_optimizers(self.model, self.cfg.optimization, max_steps=self.cfg.training["max_steps"] * 2, num_processes=self.accelerator.num_processes)

    def _dataloader(self):
        return eu.create_dataloader(self)

    def get_test_loader(self):
        self.test_loader =  eu.create_test_dataloader(self)
    

    def set_objective_and_metrics(self) -> None:
        self.objective_name = self.cfg.training["objective"]
        self.objective = partial(OBJECTIVES[self.objective_name], pad_id = self.cfg.model.get("pad_id", 0))
        # count the number of parameters
        eu.set_metrics(self, objective=self.objective_name, istraining=self.cfg.istraining)

    def set_tokenizer(self) -> None:
        # configure text tokenizer
        TextTokenzier =  AutoTokenizer.from_pretrained(resolve_pretrained_path(self.cfg.model["lm"]), cache_dir=hf_cache_dir())
        self.text_tokenizer = TextTokenzier
        # load tokenizer 
        self.tokenizer = EsmTokenizer.from_pretrained(resolve_pretrained_path(self.cfg.model["tokenizer"]), cache_dir=hf_cache_dir())

    def load_state_from_checkpoints(self, checkpoint_path, map_location=None) -> None:
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device if map_location is None else map_location))
        # from collections import OrderedDict
        # checkpoints = torch.load(checkpoint_path, map_location=self.accelerator.device)
        # self.model.load_state_dict(OrderedDict({k.replace("module.", ""): v for k, v in checkpoints["state_dict"].items()}))


    @torch.no_grad()
    def start_sampling(self) -> None:
        if self.cfg.training.objective == "next_token_prediction" :
            sample_fn = sample.conditional_sample_aa
        else:
            saprot_tokenizer, saprot = eu.load_SaProt(self.cfg.sample.saprot_path, device=self.accelerator.device)
            sample_fn = partial(sample.conditional_sample_structure_token, saprot=saprot, saprot_tokenizer=saprot_tokenizer)
        
        res_seq_dict = {}
        for idx, batch in tqdm(
            enumerate(self.sample_dataloader), total=len(self.sample_dataloader), desc=f"sampling...", disable=not self.accelerator.is_main_process
        ):
            idx_ = idx * self.accelerator.num_processes + dist.get_rank()
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to("cuda")
            seq_dict = sample_fn(
                model=self.model,
                tokenizer=self.tokenizer, 
                sample_cfg=self.cfg.sample,
                batch=batch,
                batch_idx=idx_,
                # outdir=self.cfg.io['outdir'],
                num_samples_per_condition=self.cfg.sample["n_samples"], 
            )
            res_seq_dict.update(seq_dict)
        self.accelerator.wait_for_everyone()
        gathered_seq_dicts = gather_object([res_seq_dict])
        if self.accelerator.is_main_process:
            print(f"results writing to {os.path.join(self.cfg.io['outdir'], 'seq.fasta')}")
            all_seq_dict = {}
            for seq_dict in gathered_seq_dicts:
                all_seq_dict.update(seq_dict)
            with open(os.path.join(self.cfg.io['outdir'], "seq.fasta"), 'w') as f:
                for seq_name, seq in all_seq_dict.items():
                    f.write(f">{seq_name}\n{seq}\n")
        self.accelerator.wait_for_everyone()
        self._log.info("Sample complete.")
