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
@inproceedings{OrtizSuarezSagotRomary2019,
  author    = {Pedro Javier {Ortiz Su{'a}rez} and Benoit Sagot and Laurent Romary},
  title     = {Asynchronous pipelines for processing huge corpora on medium to low resource infrastructures},
  series = {Proceedings of the Workshop on Challenges in the Management of Large Corpora (CMLC-7) 2019. Cardiff, 22nd July 2019},
  editor    = {Piotr Bański and Adrien Barbaresi and Hanno Biber and Evelyn Breiteneder and Simon Clematide and Marc Kupietz and Harald L{"u}ngen and Caroline Iliadi},
  publisher = {Leibniz-Institut f{"u}r Deutsche Sprache},
  address   = {Mannheim},
  doi       = {10.14618/ids-pub-9021},
  url       = {http://nbn-resolving.de/urn:nbn:de:bsz:mh39-90215},
  pages     = {9 -- 16},
  year      = {2019},
  abstract  = {Common Crawl is a considerably large, heterogeneous multilingual corpus comprised of crawled documents from the internet, surpassing 20TB of data and distributed as a set of more than 50 thousand plain text files where each contains many documents written in a wide variety of languages. Even though each document has a metadata block associated to it, this data lacks any information about the language in which each document is written, making it extremely difficult to use Common Crawl for monolingual applications. We propose a general, highly parallel, multithreaded pipeline to clean and classify Common Crawl by language; we specifically design it so that it runs efficiently on medium to low resource infrastructures where I/O speeds are the main constraint. We develop the pipeline so that it can be easily reapplied to any kind of heterogeneous corpus and so that it can be parameterised to a wide range of infrastructures. We also distribute a 6.3TB version of Common Crawl, filtered, classified by language, shuffled at line level in order to avoid copyright issues, and ready to be used for NLP applications.},
  language  = {en}
}
"""

_DESCRIPTION = """\
This is a small subset representing 10K records from the original OSCAR dataset, "unshuffled_deduplicated_en" subset - created for testing. The records were extracted after having been shuffled.

The full 1TB+ dataset is at https://huggingface.co/datasets/oscar.
"""

_URL = "https://cdn-datasets.huggingface.co/nlp/datasets/oscar/oscar-en-10k.tar.xz"

class OscarEn10k(datasets.GeneratorBasedBuilder):
    """The OSCAR dataset."""

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
            homepage="https://oscar-corpus.com/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        dl_dir = dl_manager.download_and_extract(_URL)
        jsonl_file = os.path.join(dl_dir, "oscar-en-10k", "oscar-en-10k.jsonl")
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
