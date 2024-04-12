#!/usr/bin/env python

# This script creates a smallish random model, with a few layers to test things quickly
#
# It also demonstrates how to change the config in child objects of the model config
#
# It will be used then as "stas/idefics-tiny-random"

from transformers import AutoTokenizer, IdeficsConfig, IdeficsForVisionText2Text

mname_from = "HuggingFaceM4/idefics-9b"
mname_very_small = "idefics-tiny-random"

tokenizer = AutoTokenizer.from_pretrained(mname_from)
config = IdeficsConfig.from_pretrained(mname_from)

config.update(dict(
    hidden_size=64,
    intermediate_size=37,
    num_hidden_layers=5,
    num_attention_heads=4,
    max_position_embeddings=64,
    max_sequence_length=64,

))

# This model contains several child config objects
#
# If you need to update the child config objects you can't do it from the top-level dict, but need
# to update these directly via those objects, like so:
config.perceiver_config.update(dict(qk_layer_norms_perceiver=False))
config.vision_config.update(dict(embed_dim=64))

print("new config", config)

very_small_model = IdeficsForVisionText2Text(config)
print(f"num of params {very_small_model.num_parameters()}")
very_small_model.resize_token_embeddings(len(tokenizer))

# Save
very_small_model.bfloat16() # makes it smaller
very_small_model.save_pretrained(mname_very_small)
config.save_pretrained(mname_very_small)
tokenizer.save_pretrained(mname_very_small)

print(f"Generated {mname_very_small}")

# Upload
# transformers-cli repo create idefics-tiny-random
# clone and add files
