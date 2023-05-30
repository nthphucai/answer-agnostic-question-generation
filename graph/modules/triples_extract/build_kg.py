import numpy as np
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer

import networkx as nx
from graph.modules.triples_extract.extract_features import Processor
from graph.modules.triples_extract.extract_triples import TripleExtractor
from graph.modules.triples_extract.utils import convert_to_geometric
from questgen.utils.file_utils import load_json_file
from questgen.utils.utils import set_gpu
from torch_geometric.utils import from_networkx


class KnowledgeGraphExtractor(Processor):
    def __init__(self, lang="en", type="spacy", tokenizer=None, model=None):
        super().__init__(tokenizer=tokenizer, model=model)
        self.extractor = TripleExtractor(lang=lang, type=type)

    def create_geometric_instance(self, kg_df, examples: list):
        current_kg_df, geometric_data = self.build_kg_from_data(sentences=examples)
        graph_src, graph_tgt, graph_rel = self.mapping_to_kg(
            base_df=kg_df, current_kg_df=current_kg_df
        )
        nodes_features, tgt_features, rel_features = self.convert_to_feature(
            graph_src, graph_tgt, graph_rel
        )
        G = nx.from_pandas_edgelist(
            df=current_kg_df,
            source="source",
            target="target",
            edge_attr=True,
            create_using=nx.DiGraph(),
        )

        data = from_networkx(G)
        data.edge_relation = rel_features
        data.node_features = nodes_features
        return data

    def convert_to_feature(self, graph_src, graph_tgt, graph_rel):
        nodes_features = self.process(
            graph_src["source"], aggre=({"use": False, "axis": 1})
        )
        tgt_features = self.process(
            graph_tgt["target"], aggre=({"use": False, "axis": 1})
        )
        rel_features = self.process(
            graph_rel["relation"], aggre=({"use": True, "axis": 1})
        )
        return nodes_features, tgt_features, rel_features

    def build_kg_from_data(self, sentences):
        kg_df = self.extractor.extract(context_list=sentences, return_df=True)
        kg_df, geometric_data = convert_to_geometric(kg_df, return_df=True)
        new_df = pd.DataFrame(
            {
                "source": kg_df["target"],
                "target": kg_df["source"],
                "relation": kg_df["relation"],
            }
        )
        new_df = new_df.drop_duplicates()
        return new_df, geometric_data

    def mapping_to_kg(self, base_df, current_kg_df):
        # mapping_src = base_df[base_df["source"].isin(current_kg_df['source'].values)].drop_duplicates().reset_index(drop=True)
        # mapping_tgt = base_df[base_df["target"].isin(current_kg_df['target'].values)].drop_duplicates().reset_index(drop=True)
        # mapping_rel = base_df[base_df["relation"].isin(current_kg_df['relation'].values)].drop_duplicates().reset_index(drop=True)
        return current_kg_df


if __name__ == "__main__":
    set_gpu(1)

    lm_model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    data = load_json_file(
        "/home/phucnth/questgen/data/simple-question/english/mcqg_english_data_1000_v1.1.0.json"
    )["data"]
    base_sentences = np.unique([c["context"] for c in data]).tolist()

    extractor = KnowledgeGraphExtractor(lang="en", type="corenlp")
    base_kg, _ = extractor.build_kg_from_data(sentences=base_sentences[:1])
    # base_kg = pd.read_csv("graph/kg_df.csv", index_col=0)
    print(base_kg)
