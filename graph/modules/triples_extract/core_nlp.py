import os
import re
from typing import List

from dotenv import load_dotenv
from stanfordnlp.server import CoreNLPClient


load_dotenv()
CORENLP_HOME = os.getenv("CORENLP_HOME")
os.environ["CORENLP_HOME"] = CORENLP_HOME
properties = {"openie.affinity_probability_cap": 2 / 3}


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class CoreNLP(metaclass=Singleton):
    def __init__(self, annotators: List[str] = None, *args, **kwargs):
        if not annotators:
            annotators = [
                "tokenize",
                "ssplit",
                "pos",
                "lemma",
                "ner",
                "parse",
                "dcoref",
                "openie",
            ]
        self.client = CoreNLPClient(annotators=annotators, memory="8G", *args, **kwargs)

    def annotate(
        self,
        text: str,
        annotators: List[str],
        properties_key: str = None,
        properties: dict = None,
        simple_format_triples: bool = True,
    ):
        core_nlp_output = self.client.annotate(
            text=text,
            annotators=annotators,
            output_format="json",
            properties_key=properties_key,
            properties=properties,
        )
        self.collect_log()
        if simple_format_triples and annotators == ["openie"]:
            triples = []
            for sentence in core_nlp_output["sentences"]:
                for triple in sentence["openie"]:
                    triples.append(
                        {
                            "subject": triple["subject"],
                            "relation": triple["relation"],
                            "object": triple["object"],
                        }
                    )
            return triples
        else:
            return core_nlp_output

    def __del__(self):
        if hasattr(self, "client"):
            self.client.stop()
        self.collect_log()
        del os.environ["CORENLP_HOME"]

    @staticmethod
    def collect_log():
        execute_path = os.environ["PYTHONPATH"].split(":")[0]
        for f in os.listdir(execute_path):
            if re.search(r"corenlp_server-.*(?:props)", f):
                os.remove(os.path.join(execute_path, f))
