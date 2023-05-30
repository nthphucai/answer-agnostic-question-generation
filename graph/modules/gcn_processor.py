import re
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import networkx as nx
from torch_geometric.utils import from_networkx


NUM_CHOICES = 4


class GCNProcessor:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model.encoder

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def process(self, dataset):
        dataset = dataset.map(self.preprocess)
        dataset = dataset.map(self.create_geometric_instance_one_item)
        dataset = dataset.map(self.mapping_feature_one_item)
        return dataset

    def preprocess(self, examples: dict):
        """
        Process and clean the values in the 'source_subject', 'source_tgt', and 'source_rel' fields of the 'examples' dictionary.

        Parameters:
        - examples (dict): A dictionary containing fields 'source_subject', 'source_tgt', and 'source_rel' to be processed.

        Returns:
        - None

        The function performs the following operations on the specified fields of the 'examples' dictionary:
        1. Removes single quotes, square brackets, and periods from the values.
        2. Splits the modified strings into lists using commas as the delimiter.
        3. Strips leading and trailing whitespace from each element in the lists.
        4. Truncates the lists to a specified length, defined by the 'NUM_CHOICES' constant.

        Note: The modified lists are reassigned to the corresponding keys in the 'examples' dictionary.
        """

        examples["source_subject"] = re.sub(
            "['\[\].]", "", examples["source_subject"]
        ).split(",")
        examples["source_subject"] = [
            example.strip() for example in examples["source_subject"]
        ]
        examples["source_subject"] = examples["source_subject"]

        examples["source_tgt"] = re.sub("['\[\].]", "", examples["source_tgt"]).split(
            ","
        )
        examples["source_tgt"] = [example.strip() for example in examples["source_tgt"]]
        examples["source_tgt"] = examples["source_tgt"]

        examples["source_rel"] = re.sub("['\[\].]", "", examples["source_rel"]).split(
            ","
        )
        examples["source_rel"] = [example.strip() for example in examples["source_rel"]]
        examples["source_rel"] = examples["source_rel"]
        return examples

    def create_geometric_instance_one_item(self, examples):
        """
        Creates a PyTorch Geometric `GeometricData` instance for a single item (subject, relation, or target) in a dataset.

        Args:
            examples (Dict[str]): A dictionary containing the string representations of the subject, relation, and target for the item.

        Returns:
            A dictionary containing the modified input `examples` dictionary with additional attributes for the `GeometricData` instance:

            - "edge_index" List of (torch.Tensor(2, num_relation)): The edge index of the `GeometricData` instance.
            - "source_rel" List of (torch.Tensor(num_relation)): The source relation of the `GeometricData` instance.
            - "source_subject" List of (num_choices, node_dim) : The source subject.
            - "source_tgt" List of (num_choices, node_dim): The source target.
        """
        source_subject = examples["source_subject"]
        source_tgt = examples["source_tgt"]
        source_rel = examples["source_rel"]

        MIN_LENGTH = min([len(source_subject), len(source_tgt), len(source_rel)])

        kg_df = pd.DataFrame(
            {
                "source_subject": source_subject[:MIN_LENGTH],
                "source_tgt": source_tgt[:MIN_LENGTH],
                "source_rel": source_rel[:MIN_LENGTH],
            }
        )
        G = nx.from_pandas_edgelist(
            df=kg_df,
            source="source_subject",
            target="source_tgt",
            edge_attr=True,
            create_using=nx.MultiDiGraph(),
        )
        nodes = tuple(G.nodes())
        data = from_networkx(G)

        examples["edge_index"] = data.edge_index[:, :NUM_CHOICES]
        examples["source_rel"] = data.source_rel[:NUM_CHOICES]
        examples["source_subject"] = [nodes[idc] for idc in examples["edge_index"][0]]
        examples["source_tgt"] = [nodes[idc] for idc in examples["edge_index"][1]]

        loop_index = examples["edge_index"][0]
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        examples["edge_index"] = torch.cat([examples["edge_index"], loop_index], dim=1)
        examples["source_rel"] = examples["source_rel"] + [
            nodes[idc] for idc in loop_index[0]
        ]
        examples["source_subject"] = [nodes[idc] for idc in examples["edge_index"][0]]
        return examples

    def extract_features_one_item(
        self, examples: List[str], aggre=({"use": False, "axis": 1})
    ) -> np.ndarray:
        """Extracts features for item in subject/relation/target from a list of string representations for dataset in Pyarrow.

        Args:
            examples (List[str]): The list of string represents a subject/relation/target.
            aggre (Tuple, optional): A tuple specifying whether to use an aggregate function for the features.
            Defaults to use mean ({'use':False, 'axis': 1}).

        Returns:
            An array containing the extracted features for the item, with the following shapes:

            - Subject features -> (num_choices, node_dimension)
            - Target features -> (num_choices, node_dimension)
            - Relation features -> (num_relation,)

        """
        tokens = list(map(self.apply_tokens, examples))
        tokens = DataLoader(tokens, batch_size=1, collate_fn=self.collate_fn)

        temp_lst = []
        for _, batch in enumerate(iterable=tokens):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            features = self.model(input_ids, attention_mask)
            features = features.last_hidden_state
            temp_lst.append(features.detach().cpu().numpy())

        features = np.concatenate(temp_lst)
        features = features.mean(axis=aggre["axis"]) if aggre["use"] else features
        return features

    def mapping_feature_one_item(self, examples):
        examples["source_subject"] = self.extract_features_one_item(
            examples["source_subject"], aggre=({"use": True, "axis": 1})
        )
        examples["source_tgt"] = self.extract_features_one_item(
            examples["source_tgt"], aggre=({"use": True, "axis": 1})
        )
        examples["source_rel"] = self.extract_features_one_item(
            examples["source_rel"], aggre=({"use": True, "axis": (1, 2)})
        )
        return examples

    def apply_tokens(self, example_batch):
        encodings = self.tokenizer.encode_plus(
            example_batch,
            max_length=32,
            padding="max_length",
            add_special_tokens=True,
            pad_to_max_length=True,
            truncation=True,
        )
        encodings = {
            "input_ids": torch.tensor(encodings["input_ids"]),
            "attention_mask": torch.tensor(encodings["attention_mask"]),
        }
        return encodings

    @staticmethod
    def collate_fn(example_batch: list) -> dict:
        from torch.nn.utils.rnn import pad_sequence

        input_ids = pad_sequence(
            [example["input_ids"] for example in example_batch], batch_first=True
        )
        attention_mask = pad_sequence(
            [example["attention_mask"] for example in example_batch], batch_first=True
        )
        batch = {"input_ids": input_ids, "attention_mask": attention_mask}
        return batch
