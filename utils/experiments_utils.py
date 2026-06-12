import os
import torch 
import pickle
from dataloader.datamodules.multitask_datamodule import MTDataModule
from utils.metrics import Accuracy, Scalar
from torchmetrics import Metric
from transformers import (get_cosine_schedule_with_warmup,
                          get_polynomial_decay_schedule_with_warmup, 
                          AutoTokenizer)
from torch.utils.data import DataLoader
import shutil
from omegaconf import OmegaConf
from collections import OrderedDict
from transformers import EsmTokenizer
from models import SaProtIFModel
import torch.distributed as dist
from dataloader.datasets import TextDataset
from transformers import EsmForMaskedLM
from utils.path_utils import hf_cache_dir, resolve_pretrained_path


mask_strategy_dict = {
    "woAA-partialstructure": 0,
    "woAA-fullstructure":  1,
    "partialAA-wostructure": 2,
    "partialAA-partialstructure": 3,
    "partialAA-fullstructure": 4,
}


def load_SaProt(saprot_path, device):
    saprot_path = resolve_pretrained_path(saprot_path)
    SaProt_tokenizer = EsmTokenizer.from_pretrained(saprot_path, cache_dir=hf_cache_dir())
    SaProt = EsmForMaskedLM.from_pretrained(saprot_path, cache_dir=hf_cache_dir())
    SaProt.to(device)
    return SaProt_tokenizer, SaProt

def on_main_process(func):
    def wrapper(*args, **kwargs):
        if not dist.is_initialized() or dist.get_rank() == 0: # Check if this is the main process
            return func(*args, **kwargs)
        else:
            return None
    return wrapper

def get_date():
    import datetime
    return datetime.datetime.now().strftime("%m-%d")


def calculate_accuracy(logits, target):
    preds = logits.argmax(dim=-1)
    preds = preds[target != -100]
    target = target[target != -100]
    if target.numel() == 0:
        return 1
    assert preds.shape == target.shape
    correct = torch.sum(preds == target)
    total = target.numel()
    return correct / total


def create_sample_dataloader(dataset_cfg, sample_cfg, text_tokenizer):
    if sample_cfg["test"]:
        if sample_cfg["long_text"]:
            data_path = dataset_cfg["text_dataset_path_for_generation"]
        else:
            data_path = dataset_cfg["text_dataset_path_for_generation_short"]
    else:
        data_path = dataset_cfg["text_dataset_path_for_generation_val"]
    dataset = TextDataset(
        data_path=data_path,
        sprot_data_path=dataset_cfg["sprot_data_path"],
        text_tokenizer=text_tokenizer,
        text_max_sequence_len=dataset_cfg["max_text_seq_len"],
        skip_batches=dataset_cfg.get("skip_batches", 0),
    )
    return DataLoader(
        dataset, batch_size=dataset_cfg.get("batch_size", 1), shuffle=False, num_workers=8
    )

def wapper_save_top_3_checkpoint(exp, step, loss):
    # check if is deepspeed zero3
    if exp.zero_stage == 3:
        save_top_3_zero3_checkpoint(exp, step, loss)
    else:
        save_top_3_checkpoint(exp, step, loss)

def save_top_3_zero3_checkpoint(exp, step, loss):
    output_dir = os.path.join(exp.cfg.io["outdir"], 'BestCheckpoints',  f"step={step}_loss={round(loss, 4)}")
    exp.model.save_checkpoint(output_dir, "pytorch_model")
    exp._log.info(f"DeepSpeed Model and Optimizer saved to output dir {os.path.join(output_dir, 'pytorch_model')}")

@on_main_process
def save_top_3_checkpoint(exp, step, loss):
    output_dir = os.path.join(exp.cfg.io["outdir"], 'BestCheckpoints')
    output_path = os.path.join(output_dir, f"step={step}_loss={round(loss, 4)}.pth")
    exp._log.info(f"saving checkpoint at step {step} to {output_path} for the top-3 metric {loss}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    write_checkpoint(ckpt_path=output_path, state_dict=exp.model.state_dict(), conf=exp.cfg, optimizer=None, 
                     epoch=exp.trained_epochs, step=step, lr_scheduler=exp.scheduler.state_dict())


@on_main_process
def save_checkpoint_with_step(exp, step):
    output_dir = os.path.join(exp.cfg.io["outdir"], 'IntervalCheckpoints')
    output_path = os.path.join(output_dir, f"step={step}.pth")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    exp._log.info(f"saving checkpoint at step {step} to {output_path}")
    write_checkpoint(ckpt_path=output_path, state_dict=exp.model.state_dict(), conf=exp.cfg, optimizer=exp.optimizer.state_dict(), 
                     epoch=exp.trained_epochs, step=step, lr_scheduler=exp.scheduler.state_dict())
    

def set_metrics(exp, objective="default", istraining=True):
    if istraining:
        setattr(exp, f"train_{objective}_accuracy", Accuracy().to(exp.device))
        setattr(exp, f"train_{objective}_loss", Scalar().to(exp.device))
        setattr(exp, f"val_{objective}_accuracy", Accuracy().to(exp.device))
        setattr(exp, f"val_{objective}_loss", Scalar().to(exp.device))
    else:
        setattr(exp, f"test_{objective}_accuracy", Accuracy().to(exp.device))
        setattr(exp, f"test_{objective}_loss", Scalar().to(exp.device))

def set_metrics_acc(exp, objective="default", istraining=True):
    if istraining:
        setattr(exp, f"train_{objective}_accuracy", Accuracy().to(exp.device))
        setattr(exp, f"val_{objective}_accuracy", Accuracy().to(exp.device))
    else:
        setattr(exp, f"test_{objective}_accuracy", Accuracy().to(exp.device))


def args_to_dict(args, **kwargs):
    """Convert argparse.Namespace to dict."""
    res_dic = {}
    for k, v in vars(args).items():
        if isinstance(v, dict):
            for k1, v1 in v.items():
                res_dic[f"{k}_{k1}"] = v1
        else:
            res_dic[k] = v
    for k, v in kwargs.items():
        res_dic[k] = v
    return  res_dic

def create_test_dataloader(exp):
    from dataloader.datasets import CATHDataset
    return dm.test_dataloader()

def create_dataloader(exp):
    dm = MTDataModule(exp.cfg.model, exp.cfg.dataset, exp.cfg.training, istraining=exp.cfg.istraining, _log=exp._log)
    dm.setup()
    if exp.cfg.istraining:
        train_dl = dm.train_dataloader()
        if exp.cfg.dataset.hard_val_metrics:
            val_dl = [dm.hard_val_dataloader()]
        else:
            val_dl = [dm.val_dataloader()]
        return train_dl, val_dl
    else:
        return dm.test_dataloader()


def count_parameters(exp):
    # count the number of parameters
    num_parameters = sum(p.numel() for p in exp.model.parameters())
    num_trainable_parameters = sum(p.numel() for p in exp.model.parameters() if p.requires_grad)
    template_line  = "\033[1;31m{:#^50}\033[0m\n"
    info_str = "\n" + template_line.format(" Number of parameters ")
    total_para_line = " {:^30s} {:.5}M ".format("total parameters", num_parameters/1e6)
    trainable_para_line = " {:^30s} {:.5}M ".format("trainable model parameters", num_trainable_parameters/1e6)
    info_str += template_line.format(total_para_line) + template_line.format(trainable_para_line)
    for name, module in exp.model.named_children():
        if isinstance(module, Metric):
            continue
        num_parameters = sum(p.numel() for p in module.parameters())
        # exp._log.info(f"Number of {name} parameters {num_parameters / 1e6}M")
        line = ' {:^30s} {:.5}M '.format(name, num_parameters/1e6)
        info_str += template_line.format(line)
    info_str += template_line.format("")
    exp._log.info(info_str)


def configure_optimizers(model, opt_config, max_steps, num_processes):

    lr = opt_config["learning_rate"]
    plm_lr = lr
    if opt_config["plm_learning_rate"] != -1:
        plm_lr = opt_config["plm_learning_rate"]

    lm_lr = lr
    if opt_config["lm_learning_rate"] != -1:
        lm_lr = opt_config["lm_learning_rate"]

    wd = opt_config["weight_decay"]

    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight",
        "ln_1.weight",
        "ln_f.weight",
    ]
    decay_power = opt_config["decay_power"]
    optim_type = opt_config["optim_type"]

    lm_para = lambda x: "lm." in x and "plm." not in x
    plm_para = lambda x: "plm." in x
    lm_params_for_normal = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and lm_para(n) and not any(nd in n for nd in no_decay)
    ]
    plm_params_for_normal = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and plm_para(n) and not any(nd in n for nd in no_decay)
    ]
    other_params_for_normal = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and not any(nd in n for nd in no_decay) and not lm_para(n) and not plm_para(n)
    ]
    lm_params_for_nodecay = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and lm_para(n) and any(nd in n for nd in no_decay)
    ]
    plm_params_for_nodecay = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and plm_para(n) and any(nd in n for nd in no_decay)
    ]
    other_params_for_nodecay = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and any(nd in n for nd in no_decay) and not lm_para(n) and not plm_para(n)
    ]

    optimizer_grouped_parameters = [
        {
            "params": lm_params_for_normal,
            "weight_decay": wd,
            "lr": lm_lr,
        },
        {
            "params": plm_params_for_normal,
            "weight_decay": wd,
            "lr": plm_lr,
        },
        {
            "params": other_params_for_normal,
            "weight_decay": wd,
            "lr": lr,
        },
        {
            "params": lm_params_for_nodecay,
            "weight_decay": 0,
            "lr": lm_lr,
        },
        {
            "params": plm_params_for_nodecay,
            "weight_decay": 0,
            "lr": plm_lr,
        },
        {
            "params": other_params_for_nodecay,
            "weight_decay": 0,
            "lr": lr,
        },
    ]
    if optim_type == "adamw":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, eps=1e-8, betas=(0.9, 0.98)
        )
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)


    warmup_steps = opt_config["warmup_steps"]
    if isinstance(opt_config["warmup_steps"], float):
        warmup_steps = int(max_steps * warmup_steps)
    if decay_power == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps * num_processes,
            num_training_steps=max_steps * num_processes,
        )
    else:
        raise NotImplementedError

    return optimizer, scheduler


def write_pkl(
        save_path, pkl_data, create_dir = False, use_torch=False):
    """Serialize data into a pickle file."""
    if create_dir:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if use_torch:
        torch.save(pkl_data, save_path)
    else:
        with open(save_path, 'wb') as handle:
            pickle.dump(pkl_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def write_checkpoint(
        ckpt_path: str,
        state_dict,
        conf,
        optimizer=None,
        epoch=None,
        step=None,
        lr_scheduler=None,
        use_torch=True,
    ):
    """Serialize experiment state and stats to a pickle file.

    Args:
        ckpt_path: Path to save checkpoint.
        conf: Experiment configuration.
        optimizer: Optimizer state dict.
        epoch: Training epoch at time of checkpoint.
        step: Training steps at time of checkpoint.
        exp_state: Experiment state to be written to pickle.
        preds: Model predictions to be written as part of checkpoint.
    """
    write_pkl(
        save_path=ckpt_path,
        pkl_data={
            'state_dict': state_dict,
            'conf': conf,
            'optimizer': optimizer,
            'epoch': epoch,
            'step': step, 
            "lr_scheduler": lr_scheduler,
        },
        use_torch=use_torch)
