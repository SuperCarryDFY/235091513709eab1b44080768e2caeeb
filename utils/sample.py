import os

import torch
from transformers import GenerationConfig
from torch.nn.parallel import DistributedDataParallel
import torch.nn as nn

def write_sample_cond(
    sample,
    output_dir,
    name,
    verbose=False,
):
    with open(os.path.join(output_dir, name), "w") as f:
        for i, _s in enumerate(sample):
            f.write(">SEQUENCE_" + str(i) + "\n" + str(_s) + "\n")
    if verbose:
        print(f"Samples are saved to {output_dir}/{name}")

def conditional_sample_structure_token(model, tokenizer, sample_cfg, batch, batch_idx, num_samples_per_condition=5, saprot=None, saprot_tokenizer=None, mask_by_prob_value=-1e9):
    device = next(model.plm.parameters()).device
    if isinstance(model, DistributedDataParallel):
        model.plm = model.module.plm
        model.lm = model.module.lm

    assert saprot and saprot_tokenizer
    assert num_samples_per_condition % 5 == 0
    # Infer text
    if next(model.lm.parameters()).device == torch.device("cpu"):
        model.lm = model.lm.to(device)
    text_hidden_states, text_attention_mask = model.infer_text(batch)
    ## move lm to cpu to save gpu memory
    model.lm = model.lm.to(torch.device("cpu"))
    torch.cuda.empty_cache()
    # Define input
    batch_size= text_hidden_states.shape[0]
    start_id = tokenizer.cls_token_id
    stop_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    input_ids = (torch.zeros((1)) + start_id).unsqueeze(0).repeat(batch_size, 1)  # create batch dim
    input_ids = input_ids.to(torch.long)
    input_ids = input_ids.to(device)
    
    generation_config = GenerationConfig(
        temperature=sample_cfg["temperature"],
        top_k=sample_cfg["top_k"],
        top_p=sample_cfg["top_p"],
        do_sample=True,
        num_beams=sample_cfg["num_beams"],
        repetition_penalty=sample_cfg["repetition_penalty"],
        max_length=sample_cfg["max_length"],
        min_length=sample_cfg["min_length"],
        guidance_scale=sample_cfg["guidance_scale"],
    )

    all_structures_list = []
    sample_results = model.plm.generate(
        input_ids,
        do_sample=True,
        pad_token_id=pad_id,
        eos_token_id=stop_id,
        bos_token_id=start_id,
        generation_config=generation_config,
        num_return_sequences=5,
        encoder_hidden_states=text_hidden_states,
        encoder_attention_mask=text_attention_mask,
        return_dict_in_generate=True
    )
    to_list = lambda seq: [
        seq[i, ...].detach().cpu().numpy().tolist() for i in range(seq.shape[0])
    ]
    tokens = to_list(sample_results.sequences)

    structures = tokenizer.batch_decode(tokens)
    # convert structure token to saprot input 
    saprot_input = []
    for structrue in structures:
        saprot_input.append("#" + "#".join([i for i in structrue.split() if i.isalpha() or i == "#"]))
    all_structures_list.extend(saprot_input)
    inputs = saprot_tokenizer(saprot_input, return_tensors="pt", max_length=sample_cfg["max_length"], truncation=True, padding="max_length")
    inputs = {k: v.to(batch["text_ids"].device) for k, v in inputs.items()}
    with torch.no_grad():
        out = saprot(**inputs)
    logits = out.logits
    predicted_token = logits.argmax(-1)
    # get probs for each selected token 
    predicted_token_list = list(predicted_token)
    predicted_token_list = [saprot_tokenizer.convert_ids_to_tokens(i) for i in predicted_token_list]

    sequence_token_list = []
    attention_mask = inputs["attention_mask"].detach().cpu().tolist()
    for idx, predicted_token in enumerate(predicted_token_list):
        cur_list = []
        valid_len = int(sum(attention_mask[idx]))
        for token in predicted_token[1 : max(valid_len - 1, 1)]:
            if token[0] == "<":
                break
            cur_list.append(token[0])
        sequence_token_list.append(cur_list)

    sequence_token_list = ["".join(sequence_token) for sequence_token in sequence_token_list]
    seq_dict = {f"batch_idx_{batch_idx}_seq_idx_{i}" : seq for i, seq in enumerate(sequence_token_list)}

    
    return seq_dict

    
def conditional_sample_aa(model, tokenizer, sample_cfg, batch, batch_idx, num_samples_per_condition=5):
    
    if isinstance(model, DistributedDataParallel):
        model.plm = model.module.plm
        model.lm = model.module.lm

    # Infer text
    text_hidden_states, text_attention_mask = model.infer_text(batch)
    
    # Define input
    batch_size= text_hidden_states.shape[0]
    start_id = tokenizer.cls_token_id
    stop_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    input_ids = (torch.zeros((1)) + start_id).unsqueeze(0).repeat(batch_size, 1)  # create batch dim
    input_ids = input_ids.to(torch.long)
    input_ids = input_ids.to(next(model.parameters()).device)
    
    generation_config = GenerationConfig(
        temperature=sample_cfg["temperature"],
        top_k=sample_cfg["top_k"],
        top_p=sample_cfg["top_p"],
        do_sample=True,
        num_beams=sample_cfg["num_beams"],
        repetition_penalty=sample_cfg["repetition_penalty"],
        max_length=sample_cfg["max_length"],
        min_length=sample_cfg["min_length"],
        guidance_scale=sample_cfg["guidance_scale"],
    )
    all_tokens = model.plm.generate(
        input_ids,
        do_sample=True,
        pad_token_id=pad_id,
        eos_token_id=stop_id,
        bos_token_id=start_id,
        generation_config=generation_config,
        num_return_sequences=num_samples_per_condition,
        encoder_hidden_states=text_hidden_states,
        encoder_attention_mask=text_attention_mask,
    )

    to_list = lambda seq: [
        seq[i, ...].detach().cpu().numpy().tolist() for i in range(seq.shape[0])
    ]
    all_tokens = to_list(all_tokens)
    all_sequences = tokenizer.batch_decode(all_tokens, skip_special_tokens=True)
    all_sequences = [seq.replace(" ", "") for seq in all_sequences]
    seq_dict = {f"batch_idx_{batch_idx}_seq_idx_{i}" : seq for i, seq in enumerate(all_sequences)}
    return seq_dict
