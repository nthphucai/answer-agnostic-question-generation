import itertools
from abc import ABC, abstractmethod
from typing import Dict, List

import pandas as pd
import spacy
from spacy.matcher import Matcher

from graph.modules.triples_extract.core_nlp import CoreNLP
from graph.modules.triples_extract.langchain_extract_triplet import GraphExtractor
from graph.utils.visualize_utils import visualize_graph_from_corenlp
from questgen.utils.utils import get_progress


class BaseTriples(ABC):
    def __init__(self, tokenizer=None, lang="en", type="corenlp"):
        assert lang in ("en", "vi"), "lang must be en or vi"

        if lang == "en":
            self.nlp = spacy.load("en_core_web_sm")
        elif lang == "vi":
            self.nlp = None

        self.tokenizer = tokenizer

    @abstractmethod
    def extract_one_item(self) -> List[Dict[str, str]]:
        pass

    def _add_special_tokens(self, context):
        triples = self.extract_one_item(context)
        return (
            "<sep>".join(
                [
                    "[{}] {} [{}]".format(d["subject"], d["relation"], d["object"])
                    for d in triples
                ]
            )
            + "<eos>"
        )

    def convert_to_features(self, example):
        graph_inputs = self.tokenizer(
            example,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return graph_inputs

    def get_entities_spacy(self, sent):
        ## chunk 1
        ent1 = ""
        ent2 = ""

        prv_tok_dep = ""  # dependency tag of previous token in the sentence
        prv_tok_text = ""  # previous token in the sentence

        prefix = ""
        modifier = ""

        #############################################################

        for tok in self.nlp(sent):
            ## chunk 2
            # if token is a punctuation mark then move on to the next token
            if tok.dep_ != "punct":
                # check: token is a compound word or not
                if tok.dep_ == "compound":
                    prefix = tok.text
                    # if the previous word was also a 'compound' then add the current word to it
                    if prv_tok_dep == "compound":
                        prefix = prv_tok_text + " " + tok.text

            # check: token is a modifier or not
            if tok.dep_.endswith("mod") == True:
                modifier = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    modifier = prv_tok_text + " " + tok.text

            ## chunk 3
            if tok.dep_.find("subj") == True:
                ent1 = modifier + " " + prefix + " " + tok.text
                prefix = ""
                modifier = ""
                prv_tok_dep = ""
                prv_tok_text = ""

            ## chunk 4
            if tok.dep_.find("obj") == True:
                ent2 = modifier + " " + prefix + " " + tok.text

            ## chunk 5
            # update variables
            prv_tok_dep = tok.dep_
            prv_tok_text = tok.text
        #############################################################

        return [ent1.strip(), ent2.strip()]

    def get_relation_spacy(self, sent):
        doc = self.nlp(sent)

        # Matcher class object
        matcher = Matcher(self.nlp.vocab)

        # define the pattern
        pattern = [
            {"DEP": "ROOT"},
            {"DEP": "prep", "OP": "?"},
            {"DEP": "agent", "OP": "?"},
            {"POS": "ADJ", "OP": "?"},
        ]

        matcher.add(key="matching_1", patterns=[pattern])

        matches = matcher(doc)
        k = len(matches) - 1

        span = doc[matches[k][1] : matches[k][2]]

        return span.text

    def visualize(self, save_file: str = "graph.png"):
        visualize_graph_from_corenlp(self.data, png_filename=save_file)


class TripleExtractor(BaseTriples):
    def __init__(self, tokenizer=None, lang="en", type="spacy"):
        super().__init__(tokenizer=tokenizer, lang=lang, type=type)
        self.type = type

        if type == "langchain":
            self.langchain_extractor = GraphExtractor()
        elif type == "corenlp":
            self.corenlp_extractor = CoreNLP(annotators=["openie"])

    def extract(self, context_list: List[str], return_df=False, disable=False):
        triples = [
            self.extract_one_item(ctx)
            for ctx in get_progress(context_list, disable=disable)
        ]
        if type(triples[0]) is list:
            triples = list(itertools.chain(*triples))

        if return_df:
            source = [triple["subject"] for triple in triples]
            target = [triple["object"] for triple in triples]
            relations = [triple["relation"] for triple in triples]

            triples = pd.DataFrame(
                {"source": source, "target": target, "relation": relations}
            )

        return triples

    def extract_one_item(self, context: str) -> List[dict]:
        if self.type == "corenlp":
            triples = self.corenlp_extractor.annotate(
                annotators=["openie"], text=context, simple_format_triples=True
            )

        elif self.type == "spacy":
            entities = self.get_entities_spacy(context)
            relation = self.get_relation_spacy(context)
            triples = [
                {"subject": entities[0], "relation": relation, "object": entities[1]}
            ]

        elif self.type == "langchain":
            triples = self.langchain_extractor.annotate(context)

            subject = [triple[0] for triple in triples]
            object = [triple[1] for triple in triples]
            relation = [triple[2] for triple in triples]

            triples = [
                {"subject": sub, "relation": rel, "object": obj}
                for (sub, rel, obj) in zip(subject, relation, object)
            ]
        else:
            raise NotImplemented("Only support with corenlp and spacy!")

        return triples
