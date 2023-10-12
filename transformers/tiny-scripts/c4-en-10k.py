# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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
"""The Open WebText Corpus"""


import os
import json

import datasets


_CITATION = """\
@article{2019t5,
    author = {Colin Raffel and Noam Shazeer and Adam Roberts and Katherine Lee and Sharan Narang and Michael Matena and Yanqi Zhou and Wei Li and Peter J. Liu},
    title = {Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer},
    journal = {arXiv e-prints},
    year = {2019},
    archivePrefix = {arXiv},
    eprint = {1910.10683},
}
"""

_DESCRIPTION = """\
This is a small subset representing the first 10K records of the original C4 dataset, "en" subset - created for testing. The records were extracted after having been shuffled.

The full 1TB+ dataset is at https://huggingface.co/datasets/c4.
"""

_URL = "https://cdn-datasets.huggingface.co/nlp/datasets/c4/c4-en-10k.tar.xz"

class C4En10k(datasets.GeneratorBasedBuilder):
    """The C4 dataset."""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="plain_text",
            description="Plain text",
            version=datasets.Version("1.0.0"),
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({"text": datasets.Value("string")}),
            homepage="https://huggingface.co/datasets/allenai/c4/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        dl_dir = dl_manager.download_and_extract(_URL)
        jsonl_file = os.path.join(dl_dir, "c4-en-10k", "c4-en-10k.jsonl")
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"jsonl_file": jsonl_file}),
        ]

    def _generate_examples(self, jsonl_file):
        """Yields examples."""
        with open(jsonl_file, encoding="utf-8") as f:
            idx = 0
            for line in f:
                rec = json.loads(line)
                yield idx,  {"text": rec["text"]}
                idx += 1
