from collections import OrderedDict

import torch
import torch.nn as nn
from peft import LoraConfig
from transformers import (BertConfig, BertModel,
                          EsmForMaskedLM, GPT2LMHeadModel,T5EncoderModel,
                          get_cosine_schedule_with_warmup,
                          get_polynomial_decay_schedule_with_warmup)
from transformers.utils import logging
from utils.path_utils import hf_cache_dir, resolve_pretrained_path
# from models.progen2 import (ProGenConfig, ProGenForCausalLMadaLN,
#                                    ProGenForCausalLMCA)



def freeze_params(model, exclude_names=[]):
    for name, param in model.named_parameters():
        for exclude_name in exclude_names:
            if exclude_name in name:
                break
        else:
            param.requires_grad = False
            continue
    for name, module in model.named_modules():
        for exclude_name in exclude_names:
            if exclude_name in name:
                break
        else:
            module.training = False
            continue


def load_text_encoder(model_config, need_text_pooler=False, logger=None):
    if logger == None:
        logger = logging.get_logger(__name__)

    assert model_config["lm"], "lm in config should be specific."
    assert model_config["lm_init_from_pretrained"], "you should use pretrained lm."
    lm = eval(model_config["lm_type"]).from_pretrained(resolve_pretrained_path(model_config["lm"]), cache_dir=hf_cache_dir(), ignore_mismatched_sizes=True)


    try:
        model_config["lm_emb_dim"] = lm.config.hidden_size
    except:
        model_config["lm_emb_dim"] = lm.config.d_model
    
    if model_config["lm_learn_type"] == "full":
        if model_config["lm_type"] == "BertModel" and not need_text_pooler: # Since Only BertModel have pooler layer
            # This model use cross attention, which does not rely on the pooler.
            freeze_params(lm.pooler)
        # Freeze the cls head which does not contributes in loss.
        # self.freeze_params(lm.cls)
    elif model_config["lm_learn_type"] == "freeze":
        logger.info("Freezing PubMedBERT")
        freeze_params(lm)
    else:
        raise NotImplementedError
    return lm



def _init_zero_weights_linear(module):
    assert isinstance(module, nn.Linear)
    """Initialize the weights."""
    nn.init.constant_(module.weight, 0)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)
