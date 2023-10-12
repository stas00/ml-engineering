#!/usr/bin/env python
# coding: utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script creates a smallish random model, with a few layers to test things like MP/PP, where
# tiny and tiner models are too too small
#
# It will be used then as "stas/mt5-tiny-random"

# To build:
# 1. clone sentencepiece into this dir
# git clone https://github.com/google/sentencepiece
#
# 2. run this script

from pathlib import Path
import json
import tempfile

from transformers import MT5Tokenizer, MT5TokenizerFast, MT5Config, MT5ForConditionalGeneration
from transformers.models.t5.tokenization_t5 import VOCAB_FILES_NAMES

mname_from = "google/mt5-small"
mname_very_small = "mt5-tiny-random"

tokenizer = MT5Tokenizer.from_pretrained(mname_from)
config = MT5Config.from_pretrained(mname_from)
#tokenizer_fast = MT5TokenizerFast.from_pretrained(mname_from)

# Shrink the vocab of mt5-small
import sys
# HACK: need the sentencepiece source to get sentencepiece_model_pb2, as it doesn't get installed
sys.path.append("./sentencepiece/python/src/sentencepiece")
import sentencepiece_model_pb2 as model

tmp_dir = "/tmp/mt5-small"
tokenizer.save_pretrained(tmp_dir)
file = tmp_dir + "/spiece.model"
with open(file, 'rb') as f: data = f.read()

# adapted from https://blog.ceshine.net/post/trim-down-sentencepiece-vocabulary/
m = model.ModelProto()
m.ParseFromString(data)

keep_items = 5000

print("Shrinking vocab")
print(f"original dict {len(m.pieces)}")
for i in range(len(m.pieces)-keep_items): _ = m.pieces.pop()
print(f"new dict {len(m.pieces)}")

with open(tmp_dir + "/spiece-short.model", 'wb') as f:
    f.write(m.SerializeToString())

tokenizer = MT5Tokenizer(vocab_file=tmp_dir + "/spiece-short.model")

config.update(dict(
    vocab_size=keep_items+12,
    d_model=64,
    d_ff=256,
    d_kv=8,
    num_layers=8,
    num_decoder_layers=8,
    num_heads=4,
    relative_attention_num_buckets=32,
))
print("new config", config)

very_small_model = MT5ForConditionalGeneration(config)
print(f"num of params {very_small_model.num_parameters()}")
very_small_model.resize_token_embeddings(len(tokenizer))

# Test
src_texts = ["A long paragraph for summarization.", "Another paragraph for summarization."]
tgt_texts = ["Summary of the text.", "Another summary."]

batch = tokenizer.prepare_seq2seq_batch(src_texts, tgt_texts, return_tensors="pt")
outputs = very_small_model(**batch)

print("test output:", len(outputs.logits[0]))

# Save
very_small_model.half() # makes it smaller
very_small_model.save_pretrained(mname_very_small)
config.save_pretrained(mname_very_small)
tokenizer.save_pretrained(mname_very_small)
#tokenizer_fast.save_pretrained(mname_very_small)

print(f"Generated {mname_very_small}")

# Upload
# transformers-cli repo create mt5-tiny-random
# clone and add files
