import random
import torch
from  models import ABSmodule
from models.esm2 import EsmForMaskedLM, EsmForMaskedLM_proj4096
from transformers import BertModel, T5EncoderModel
from collections import OrderedDict
from transformers.models.bert.modeling_bert import BertPooler
from utils.model_utils import  freeze_params
from utils.path_utils import hf_cache_dir, resolve_pretrained_path
from easydict import EasyDict
class NullLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def exception(self, *args, **kwargs):
        pass

class SaProtIFModel(ABSmodule):
    def __init__(self, model_config, logger=None, load_pretrain=True):
        super().__init__()
        # self.save_hyperparameters()
        self.config = model_config
        if not logger:
            logger = NullLogger()
        # ===================== Text Encoder ===================== #
        self.lm = eval(model_config["lm_type"]).from_pretrained(resolve_pretrained_path(model_config["lm"]), cache_dir=hf_cache_dir(), ignore_mismatched_sizes=True)
        
        if load_pretrain and model_config.get("lm_pretrain_path", None):
            if "StructureTokenPrediction" in model_config["lm_pretrain_path"]:
                # load from trained model in stage 1.
                ckpt_lm = torch.load(model_config["lm_pretrain_path"], map_location='cpu')
                lm_state_dict = OrderedDict({
                        k[len("module.lm.") :]: v
                        for k, v in ckpt_lm["state_dict"].items()
                        if k.startswith("module.lm.")
                    })
                self.lm.load_state_dict(lm_state_dict, strict=True)
            else:
                ckpt_lm = torch.load(model_config["lm_pretrain_path"], map_location='cpu')
                self.lm.load_state_dict(
                    OrderedDict(
                        {
                            k[len("bert.") :]: v
                            for k, v in ckpt_lm.items()
                        }
                    ),
                    strict=False,
                )
        if model_config.get("load_pretrained_t5encoder", None):
            ckpt_t5encoder = torch.load(model_config["load_pretrained_t5encoder"], map_location='cpu')
            self.lm.load_state_dict(ckpt_t5encoder, strict=True)
            logger.info(f"Loaded pretrained T5 encoder from {model_config['load_pretrained_t5encoder']}")
        if model_config["lm_learn_type"] == "full":
            logger.info("Full learning Text Encoder")
        elif model_config["lm_learn_type"] == "freeze":
            logger.info("Freezing Text Encoder")
            freeze_params(self.lm, exclude_names=["pooler"])
        else:
            raise NotImplementedError
        # ===================== Protein Sequence Encoder With Textual Adapter ===================== #
        # self.plm = load_protein_model(model_config, logger=logger)
        plm_path = resolve_pretrained_path(model_config["plm_path"])
        if model_config["lm_type"] == "T5EncoderModel":
            self.plm = EsmForMaskedLM_proj4096.from_pretrained(plm_path, cache_dir=hf_cache_dir())
        else:
            self.plm = EsmForMaskedLM.from_pretrained(plm_path, cache_dir=hf_cache_dir()) # SaProt
        ###
        # freeze unused parameter 
        for para in self.plm.esm.contact_head.parameters():
            para.requires_grad = False
        for para in self.plm.esm.embeddings.position_embeddings.parameters():
            para.requires_grad = False   
        # self.plm.gradient_checkpointing_enable()
        ##
        if load_pretrain and model_config.get("saprot_pretrain_path", None):
            ckpt = torch.load(model_config["saprot_pretrain_path"], map_location='cpu')
            self.plm.load_state_dict(ckpt["model"], strict=False)
        if model_config["plm_learn_type"] == "freeze":
            freeze_params(self.plm, exclude_names=["projector"])
        elif model_config["plm_learn_type"] == "full":
            pass
        else:
            raise NotImplementedError
        # ===================== Others ===================== #
        self.partial_on_text = model_config["partial_on_text"]
        if self.config["lm_type"] == "T5EncoderModel":
            config = EasyDict({"hidden_size": 4096})
            self.pooler = BertPooler(config)
        else:
            self.pooler = None


    def infer(
        self,
        batch,
        text_hidden_states=None,
        text_attention_mask=None,
        return_dict=True,
    ):
        # concat the embeddings of text and token 
        if text_hidden_states.shape[0] != batch["mask_prot_ids"].shape[0]:
            text_hidden_states = text_hidden_states.repeat(batch["mask_prot_ids"].shape[0], 1)
        outputs = self.plm(
            input_ids=batch["mask_prot_ids"],
            attention_mask=batch["prot_masks"],
            text_token_embeddings=text_hidden_states,
            labels=batch["prot_ids"],
            return_dict=return_dict,
            output_hidden_states=True,
        )

        return {"outputs": outputs}

    def infer_text(
        self,
        batch,
    ):
        text_output = self.lm(
            input_ids=batch["text_ids"],
            attention_mask=batch["text_masks"],
            output_hidden_states=True,
        )
        # return text_output.pooler_output, batch["text_masks"]
        if self.pooler:
            return self.pooler(text_output.last_hidden_state)
        else:
            return text_output.pooler_output


    def forward(self, batch):
        ret = dict()
        if self.lm and random.random() < self.partial_on_text:
            ret["text_hidden_states"] = self.infer_text(batch)
        else:
            ret["text_hidden_states"] = None
        ret.update(
            self.infer(
                batch,
                text_hidden_states=ret["text_hidden_states"],
            )
        )
        return ret
