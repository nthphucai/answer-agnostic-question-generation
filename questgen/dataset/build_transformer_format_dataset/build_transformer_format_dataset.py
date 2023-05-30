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
"""UIT-ViQuAD: A Vietnamese Dataset for Evaluating Machine Reading Comprehension."""

import copy
import json
import logging
import random
from typing import Iterable, List, Tuple, Union

import datasets
import nltk
import numpy as np
import pandas as pd


nltk.download("punkt")

_DESCRIPTION = """\
Vietnamese Question Answering Dataset (UIT-ViQuAD) is a new dataset for the low-resource language \
as Vietnamese to evaluate MRC models. This dataset comprises over 23,000 human-generated \
question-answer pairs based on 5,109 passages of 174 Vietnamese articles from Wikipedia.
"""

_CITATION = """\
@article{2020arXiv200914725R,
    title = "{A Vietnamese Dataset for Evaluating Machine Reading Comprehension}",
    author = {Van Nguyen, Kiet and Nguyen, Duc-Vu and Nguyen, Anh Gia-Tuan and Nguyen, Ngan Luu-Thuy},
    journal = {arXiv e-prints},
    year = {2020},
    eid = {arXiv:2009.14725},
    pages = {arXiv:2009.14725},
    archivePrefix = {arXiv},
    eprint = {2009.14725},
}
"""
QG_FORMATS = ["prepend", "highlight", "prepend_highlight"]


class ViquadQGConfig(datasets.BuilderConfig):
    """BuilderConfig for ViQuAD-QG Datasets."""

    def __init__(self, qg_format="highlight", sub_task="multitask", **kwargs):
        """BuilderConfig for ViQuAD-QG.
        Args:
            **kwargs: keyword arguments forwarded to super.
        """
        super(ViquadQGConfig, self).__init__(**kwargs)
        self.qg_format = qg_format
        self.sub_task = sub_task


class ViquadQG(datasets.GeneratorBasedBuilder):
    """ViQuAD-QG: A Vietnamese Dataset for Question Generation. Version 1.1."""

    BUILDER_CONFIGS = [
        ViquadQGConfig(
            name=f"{format_}_qg_format",
            version=datasets.Version(
                "1.1.0", "New split API (https://tensorflow.org/datasets/splits)"
            ),
            description="Plain text",
            qg_format=format_,
        )
        for format_ in QG_FORMATS
    ]

    mapping_answer = {"A": 0, "B": 1, "C": 2, "D": 3}

    def _info(self):
        if "gcn" in self.config.sub_task:
            features = datasets.Features(
                {
                    "source_text": datasets.Value("string"),
                    "target_text": datasets.Value("string"),
                    "source_subject": datasets.Value("string"),
                    "source_tgt": datasets.Value("string"),
                    "source_rel": datasets.Value("string"),
                    "task": datasets.Value("string"),
                }
            )
        else:
            features = datasets.Features(
                {
                    "source_text": datasets.Value("string"),
                    "target_text": datasets.Value("string"),
                    "task": datasets.Value("string"),
                }
            )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage="https://huggingface.co/datasets/thanhns/viquad_qg/tree/main/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        urls_to_download = {
            "train": self.config.data_files["train"][0],
            "validation": self.config.data_files["validation"][0],
            "test": self.config.data_files["test"][0],
        }

        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": downloaded_files["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": downloaded_files["validation"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": downloaded_files["test"]},
            ),
        ]

    def _get_correct_alignment(self, context, gold_text, start_idx):
        """
        Some original examples in ViQuADv1.1 have indices wrong by 1 or 2 character. We test and fix this here.
        """
        end_idx = start_idx + len(gold_text)
        context, gold_text = context.lower(), gold_text.lower()

        # When the gold label position is good
        if context[start_idx:end_idx] == gold_text:
            return start_idx, end_idx

        # When the gold label is off by one character
        elif context[start_idx - 1 : end_idx - 1] == gold_text:
            return start_idx - 1, end_idx - 1

        # When the gold label is off by two character
        elif context[start_idx - 2 : end_idx - 2] == gold_text:
            return start_idx - 2, end_idx - 2

        # When the gold label is off by one character
        elif context[start_idx + 1 : end_idx + 1] == gold_text:
            return start_idx + 1, end_idx + 1

    def mapping_entities_to_kg(
        self, keywords: List[str], triples: List[str]
    ) -> Union[Tuple, List]:
        """
        Maps entities from a knowledge graph (KG) based on given keywords from question and answer.

        Args:
            keywords (List[str]): A list of keywords to search for in the KG.
            triples (List[str]): A list of triples representing the KG in the format [source, relation, target].

        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: A tuple of three pandas Series objects representing the mapped entities.
                - subject_token: A Series of source entities from the KG that match the keywords.
                - target_token: A Series of target entities from the KG that match the keywords.
                - rel_token: A Series of relation entities from the KG that match the keywords.
        """
        mapping_triples = set()
        for key in keywords:
            for _, triple in enumerate(triples):
                source_subject = triple[0]
                if key in source_subject:
                    mapping_triples.add(tuple(triple))

        mapping_triples = list(mapping_triples)
        length_mapping_triples = len(mapping_triples)
        remain_triples = [
            tuple(item) for item in triples if item not in mapping_triples
        ]

        if length_mapping_triples < 4:
            remain_triples = random.sample(remain_triples, 4 - length_mapping_triples)
            mapping_triples.extend(remain_triples)
            assert (
                len(mapping_triples) >= 4
            ), "The mapping triples length must be equal or large than 4"

        current_kg = pd.DataFrame(
            mapping_triples, columns=["source", "relation", "target"]
        )

        columns = ("source", "target", "relation")
        subject_token, target_token, rel_token = [
            pd.Series(current_kg[col]) for col in columns
        ]
        if any([len(subject_token) < 4, len(target_token) < 4, len(rel_token) < 4]):
            raise ValueError(
                "The length of subject_token, target_token, rel_token must be larger than 4"
            )

        return subject_token, target_token, rel_token

    def process_triples_text(self, triples):
        pass

    def process_distractors_text(
        self, context: str, question: Union[list, str], options: list, answers: str
    ):
        answers = answers["correct_option"]
        correct_answer_idc = self.mapping_answer[answers]
        correct_answer = options[correct_answer_idc]
        distractor_answer = [
            options[idc] for idc in range(len(options)) if idc != correct_answer_idc
        ]

        distractors_input = f"generate multiple choice: {correct_answer} {{sep_token}} {question} {{sep_token}} {context}"
        distractors_target = " ".join(
            [f"{distractor} {{sep_token}}" for distractor in distractor_answer]
        )
        return {
            "source_text": distractors_input,
            "target_text": distractors_target,
            "task": "distractors",
        }

    def process_qg_text(
        self,
        context: str,
        question: str,
        answer: dict,
        keywords: Union[List, Tuple],
        triples: List[Union[list, tuple]],
    ):
        answer_text = answer["text"][0]
        start_idx = answer["answer_start"][0]

        if self.config.qg_format == "prepend":
            question_gen_input = f"answer: {answer_text} context: {context}"
        elif self.config.qg_format == "highlight":
            start_pos, end_pos = self._get_correct_alignment(
                context, answer_text, start_idx
            )
            question_gen_input = f"generate question: {context[:start_pos]} {{hl_token}} {context[start_pos: end_pos]} {{hl_token}} {context[end_pos:]}"
        else:
            start_pos, end_pos = self._get_correct_alignment(
                context, answer_text, start_idx
            )
            question_gen_input = f"answer: {context[start_pos: end_pos]} context: {context[:start_pos]} {{hl_token}} {context[start_pos: end_pos]} {{hl_token}} {context[end_pos:]}"

        question_gen_target = f"{question}"

        examples = {
            "source_text": question_gen_input,
            "target_text": question_gen_target,
            "task": "qg",
        }
        subject_token, target_token, rel_token = self.mapping_entities_to_kg(
            keywords=keywords, triples=triples
        )

        if "gcn" in self.config.sub_task:
            examples.update(
                {
                    "source_subject": subject_token,
                    "source_tgt": target_token,
                    "source_rel": rel_token,
                }
            )

        return examples

    def process_qa_text(self, context, question, answer, keywords, triples):
        answer_text = answer["text"][0].strip()
        answer_gen_input = f"question: {question} context: {context}"
        answer_gen_target = f"{answer_text}"

        subject_token, target_token, rel_token = self.mapping_entities_to_kg(
            keywords=keywords, triples=triples
        )

        examples = {
            "source_text": answer_gen_input,
            "target_text": answer_gen_target,
            "task": "qa",
        }

        if "gcn" in self.config.sub_task:
            examples.update(
                {
                    "source_subject": subject_token,
                    "source_tgt": target_token,
                    "source_rel": rel_token,
                }
            )

        return examples

    def process_answer_extraction(self, article):
        context = article["context"].strip()
        sentences = nltk.sent_tokenize(context)
        triples = article["triples"]
        keywords = article["keywords"]

        examples = []
        source_text = "extract_answer: "
        for ans in article["answers"]["text"]:
            ans_lower = ans.lower()
            sents = copy.deepcopy(sentences)
            for idc, sent in enumerate(sents):
                sent_lower = sent.lower()
                if ans_lower in sent_lower:
                    sents[idc] = f"{{hl_token}} {sent} {{hl_token}}"

            input_text = f"{source_text}" + " ".join(sents)
            target_text = f"{ans}" + " {sep_token}"

            subject_token, target_token, rel_token = self.mapping_entities_to_kg(
                keywords=keywords, triples=triples
            )

            if "gcn" in self.config.sub_task:
                examples.append(
                    {
                        "source_text": input_text,
                        "target_text": target_text,
                        "source_subject": subject_token,
                        "source_tgt": target_token,
                        "source_rel": rel_token,
                        "task": "answer_ext",
                    }
                )
            else:
                examples.append(
                    {
                        "source_text": input_text,
                        "target_text": target_text,
                        "task": "answer_ext",
                    }
                )

        return examples

    def _generate_examples(self, filepath: str) -> Iterable:
        """
        This function returns the examples in the raw (text) form.
        """
        logging.info(f"Generating examples from {filepath}")
        count = 0

        with open(filepath) as f:
            articles = json.load(f)

        for article in articles["data"]:
            context = article["context"].strip()
            question = article["question"].strip()
            answers = article["answers"]
            triples = article["triples"]
            keywords = article["keywords"]

            if "distractors" in self.config.sub_task:
                options = article["options"]

            # Generate the examples for answer extraction task.
            if "answer_ext" in self.config.sub_task:
                answer_ext_examples = self.process_answer_extraction(article)
                for answer_ext_example in answer_ext_examples:
                    yield count, answer_ext_example
                    count += 1

            # Generate the examples for QA, QG, distractors task.
            for task in self.config.sub_task:
                if task == "distractors":
                    yield count, self.process_distractors_text(
                        context, question, options, answers
                    )
                    count += 1

                else:
                    for start_idx, answer_text in zip(
                        answers["answer_start"], answers["text"]
                    ):
                        try:
                            answer = {
                                "answer_start": [start_idx],
                                "text": [answer_text],
                            }
                            if task == "qg":
                                yield count, self.process_qg_text(
                                    context, question, answer, keywords, triples
                                )
                                count += 1

                            if task == "qa":
                                yield count, self.process_qa_text(
                                    context, question, answer, keywords, triples
                                )
                                count += 1

                        except Exception:
                            continue
