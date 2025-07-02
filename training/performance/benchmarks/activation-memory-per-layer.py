#!/usr/bin/env python

# This script derives the coefficient num_of_hidden_states_copies in `num_of_hidden_states_copies * bs * seqlen * hidden_size`, which 
rougly corresponds to the amount of hidden_states copies a given model architecture makes during a single layer's forward.

import torch
from transformers import AutoModelForCausalLM

#model_name_or_path = "Qwen/Qwen3-4B"
model_name_or_path = "google/gemma-1.1-2b-it"
#model_name_or_path = "meta-llama/Llama-3.1-8B-Instruct"
#model_name_or_path = "nvidia/Llama-3.1-Nemotron-8B-UltraLong-4M-Instruct"
#model_name_or_path = "HuggingFaceTB/SmolLM2-360M"
#model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.3"
#model_name_or_path = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.bfloat16
dtype_bytes = torch.tensor([], dtype=dtype).element_size() # 2 for bf16

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=dtype, trust_remote_code=True).to(device)

bs = 1
seqlen = 32384
hidden_size = model.config.hidden_size

hidden_states = torch.rand((bs, seqlen, hidden_size), requires_grad=True, dtype=dtype, device=device)
position_ids = torch.randint(0, seqlen, [bs, seqlen], device=device)
position_embeddings = model.model.rotary_emb(hidden_states, position_ids)

decoder_layer = model.model.layers[0]

torch.cuda.empty_cache()
before = torch.cuda.memory_allocated()
hidden_states = decoder_layer(hidden_states=hidden_states,
        attention_mask=None,
        position_ids=position_ids,
        position_embeddings=position_embeddings)
after = torch.cuda.memory_allocated()
delta = after - before

print(f'{delta / (bs * seqlen * hidden_size * dtype_bytes):.1f} "{model_name_or_path}"')
