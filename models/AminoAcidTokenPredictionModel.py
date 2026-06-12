import random
import torch.nn as nn
from utils.model_utils import load_text_encoder
from utils.path_utils import env_path, resolve_pretrained_path
from models.gpt2 import GPT2LMHeadModel, GPT2Config
class AminoAcidTokenPredictionModel(nn.Module):
    def __init__(self, model_config, logger=None):
        super().__init__()
        # self.save_hyperparameters()
        self.config = model_config
        # ===================== Text Encoder ===================== #
        self.need_text_pooler = model_config["plm_type"] == "ProGenForCausalLMadaLN"
        self.lm = load_text_encoder(model_config, need_text_pooler=self.need_text_pooler, logger=logger)

        # ===================== Protein Sequence Encoder With Textual Adapter ===================== #
        plm_config = GPT2Config.from_pretrained(resolve_pretrained_path(model_config["plm_type"]), add_cross_attention=True, 
                    cache_dir=env_path("HF_MODELS_ROOT")
                    )
        self.plm = GPT2LMHeadModel(plm_config)

        # ===================== Others ===================== #
        self.partial_on_text = model_config["partial_on_text"]

    def infer(
        self,
        batch,
        text_hidden_states=None,
        text_attention_mask=None,
        return_dict=True,
    ):
        outputs = self.plm(
            input_ids=batch["prot_ids"],
            attention_mask=batch["prot_masks"],
            encoder_hidden_states=text_hidden_states,
            encoder_attention_mask=text_attention_mask,
            labels=batch["prot_ids"],
            return_dict=return_dict,
            output_hidden_states=True,
        )

        return {"outputs": outputs}

    def infer_text(
        self,
        batch,
    ): 
        if batch["text_ids"].shape[-1]> 512:
            batch["text_ids"] = batch["text_ids"][:, :512]
            batch["text_masks"] = batch["text_masks"][:, :512]
        text_output = self.lm(
            input_ids=batch["text_ids"],
            attention_mask=batch["text_masks"],
            output_hidden_states=True,
        )
        if self.need_text_pooler:
            return text_output.pooler_output, batch["text_masks"]
        else:
            return text_output.last_hidden_state, batch["text_masks"]


    def forward(self, batch):
        ret = dict()
        if self.lm and random.random() < self.partial_on_text:
            ret["text_hidden_states"], ret["text_attention_mask"] = self.infer_text(
                batch
            )
        else:
            ret["text_hidden_states"], ret["text_attention_mask"] = None, None

        ret.update(
            self.infer(
                batch,
                text_hidden_states=ret["text_hidden_states"],
                text_attention_mask=ret["text_attention_mask"],
            )
        )
        return ret
