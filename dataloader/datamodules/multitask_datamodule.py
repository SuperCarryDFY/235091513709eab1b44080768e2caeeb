import torch.distributed as dist
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.dataset import ConcatDataset
import torch.distributed as dist
import torch 
import numpy as np 

try:
    from dataloader.datamodules import _datamodules
except:
    from dataloader.datamodules import _datamodules

# Because inital WeightedRandomSampler, the number of samples is limited to 2^24

class CustomWeightedRandomSampler(WeightedRandomSampler):
    """WeightedRandomSampler except allows for more than 2^24 samples to be sampled"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        rand_tensor = np.random.choice(range(0, len(self.weights)),
                                       size=self.num_samples,
                                       p=self.weights.numpy() / torch.sum(self.weights).numpy(),
                                       replace=self.replacement)
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())

class MTDataModule:
    def __init__(self, model_cfg, datasets_cfg, training_cfg, istraining=True, _log=None):
        
        self.datamodule_keys = datasets_cfg["datasets"]
        num_datasets = len(self.datamodule_keys)
        assert num_datasets > 0
        super().__init__()

        self.dm_keys = self.datamodule_keys
        # init data modules one by one
        self.dms = [ _datamodules[dm_key](model_cfg, datasets_cfg, training_cfg, istraining, _log) for dm_key in self.dm_keys ]
        self._log = _log
        self.mixture_rate = datasets_cfg.get("mixture_rate", [1] * len(self.dms))
        assert len(self.mixture_rate) == len(self.dms)

        # batch size and num workers is set accordings to the first dataset.
        self.batch_size = self.dms[0].batch_size
        self.num_workers_per_gpu = self.dms[0].num_workers_per_gpu
        self.per_gpu_batch_size = training_cfg["per_gpu_batchsize"]
        self.training = istraining
        self.all_num_samples= training_cfg["pre_defined_max_steps"]
    
    def setup_sampler(self):
        # num sample per epoch is set to the ratio of first dataset
        # For example, we mix up the two datasets with 1:1 (1:1 means the probability of sample comes from the two datasets is equal.), 
        # then num sample per epoch is set to 2 * lenght of the first datasets.
        weights = []
        for i in range(len(self.dms)):
            if self.dms[i].train_dataset:
                weights += self.dms[i].train_dataset.get_weights(weight=self.mixture_rate[i])
        # self.num_sample_per_epoch = len(self.dms[0].train_dataset)
        if len(self.dm_keys) > 1:
            self._log.info(f"Mixture Rate of the datsets {self.datamodule_keys}: {self.mixture_rate}")
        #     for i in range(1, len(self.dm_keys)):
        #         self.num_sample_per_epoch += int(self.mixture_rate[0] / self.mixture_rate[i]) * len(self.dms[0].train_dataset)
        self.train_sampler = CustomWeightedRandomSampler(
            weights=weights,
            # num_samples=self.all_num_samples,
            num_samples=5120000, 
            replacement=True,
        )
        self.val_sampler = None
        self.test_sampler = None


    def setup(self):
        for dm in self.dms:
            dm.setup()
        if self.training:
            self.train_dataset = ConcatDataset([dm.train_dataset for dm in self.dms if dm.train_dataset != None])
            self.val_dataset = ConcatDataset(
                [dm.val_dataset for dm in self.dms if dm.val_dataset != None]
            )
            hard_val_datasets = [dm.hard_val_dataset for dm in self.dms if getattr(dm, "hard_val_dataset", None) != None]
            if len(hard_val_datasets) > 0:
                self.hard_val_dataset = ConcatDataset(hard_val_datasets)
            else:
                self.hard_val_dataset = None
            self.collate = self.dms[0].train_dataset.collate
        else:
            self.test_dataset = ConcatDataset(
                [dm.test_dataset for dm in self.dms if dm.test_dataset != None]
            )
            self.collate = self.dms[0].test_dataset.collate
        if self.training:
            self._log.info(f"Total train: {len(self.train_dataset)}")
            self._log.info(f"Total validation: {len(self.val_dataset)}")
        else:
            self._log.info(f"Total test: {len(self.test_dataset)}")
        self.setup_sampler()
 
    def setup_train_dataset(self):
        assert self.training
        for dm in self.dms:
            dm.setup_train_dataset()
        self.train_dataset = ConcatDataset([dm.train_dataset for dm in self.dms if dm.train_dataset != None])
        self.collate = self.dms[0].train_dataset.collate
        self._log.info(f"Total train: {len(self.train_dataset)}")
        self.setup_sampler()

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers_per_gpu,
            collate_fn=self.collate,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers_per_gpu,
            collate_fn=self.collate,
        )
        return loader

    def hard_val_dataloader(self):
        loader = DataLoader(
            self.hard_val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers_per_gpu,
            collate_fn=self.collate,
        )
        return loader



    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers_per_gpu,
            collate_fn=self.collate,
        )
        return loader



if __name__ == "__main__":
    import argparse
    import os
    import torch
    from utils import config_utils
    from tqdm import tqdm
    from transformers.utils import logging
    from accelerate.utils import set_seed
    set_seed(42)


    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path for configuration file', required=True)

    args = config_utils._parse_args_and_yaml(parser)
	# Set the additional deterministic args
    args.io['run_name'] = os.path.basename(args.config).split('.')[0]
    args.GPU["num_nodes"] = int(os.environ.get("NUM_NODES", 1))
    args.GPU["num_gpus"] = torch.cuda.device_count() * args.GPU["num_nodes"]
    args.training["accumulate_grad_batches"] = args.training["pre_define_batch_size"] // (args.GPU["num_gpus"] * args.training["per_gpu_batchsize"])
    args.training["max_steps"] = args.training["pre_defined_max_steps"] // args.training["pre_define_batch_size"]
    args.training["log_every_n_steps"] = args.training["pre_defined_log_every_n_steps"] // args.training["pre_define_batch_size"]
    args.training["val_every_n_steps"] = args.training["pre_defined_val_every_n_steps"] // args.training["pre_define_batch_size"]
    args.training["save_every_n_steps"] = args.training["pre_defined_save_every_n_steps"] // args.training["pre_define_batch_size"]
    args.training["enable_top_k_steps"] = args.training["pre_defined_enable_top_k_steps"] // args.training["pre_define_batch_size"]
    args.training["num_workers"] = args.training["num_workers_per_gpu"] * args.GPU["num_gpus"]
	
	# configure output dir
    args.istraining = True
    _log = logging.get_logger(name="test")
    dm = MTDataModule(args.model, args.dataset, args.training, istraining=args.istraining, _log=_log)
    dm.setup_train_dataset()
    train_dl = dm.train_dataloader()

    data_num = 0
    text_token_length = 0

    for idx, batch in tqdm(enumerate(train_dl), total=10000):
        # print(batch["text_ids"])
        # print((batch["text_ids"] != 0).sum())
        # print((batch["text_masks"] != 0).sum())
        text_token_length += (batch["text_ids"] != 0).sum()
        data_num += len(batch["text_ids"])
        if idx > 10000:
            break

    print(f"Data num: {data_num}")
    print(f"Text token length: {text_token_length}")
    print(f"Mean text token length: {text_token_length / data_num}")
