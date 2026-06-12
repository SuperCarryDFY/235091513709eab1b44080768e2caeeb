import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

def AATokenPrediction(batch, model_out):
    outputs = model_out["outputs"]
    labels = batch["prot_ids"]
    labels = labels.masked_fill(labels == 1, -100) # cause saprot's tokenizer
    logits = outputs.logits
    loss = outputs.loss
    
    ret = {
        "loss": loss,
        "logits": logits,
        "labels": labels,
    }

    return ret


def inversefolding(batch, model_out):
    outputs = model_out["outputs"]
    labels = batch["prot_ids"]
    labels = labels.masked_fill(labels == 1, -100) # cause saprot's tokenizer
    logits = outputs.logits
    loss = outputs.loss
    
    ret = {
        "loss": loss,
        "logits": logits,
        "labels": labels,
    }

    return ret

def compute_next_token_prediction(batch, model_out, pad_id=0):
    outputs = model_out["outputs"]
    labels = batch["prot_ids"][:, 1:]
    labels = labels.masked_fill(labels == pad_id, -100)
    logits = outputs.logits[:, :-1, :]
    loss = outputs.loss
    
    ret = {
        "loss": loss,
        "logits": logits,
        "labels": labels,
    }

    return ret


def compute_next_structure_token_prediction(batch, model_out, pad_id=0):
    outputs = model_out["outputs"]
    labels = batch["structure_token_ids"][:, 1:]
    labels = labels.masked_fill(labels == pad_id, -100)
    logits = outputs.logits[:, :-1, :]
    loss = outputs.loss
    
    ret = {
        "loss": loss,
        "logits": logits,
        "labels": labels,
    }

    return ret
