from typing import List, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader

from questgen.utils.model_utils import extract_features


class Processor:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def process(
        self,
        examples: Union[pd.DataFrame, List[dict]],
        aggre=({"use": False, "axis": 1}),
    ):
        tokens = examples.apply(self.convert_to_features)
        dataloader = DataLoader(tokens, batch_size=1, collate_fn=self._generate_batch)
        features = extract_features(dataloader=dataloader, model=self.model)
        if aggre["use"]:
            features = features.mean(axis=aggre["axis"])
        return features

    def convert_to_features(self, example_batch):
        encodings = self.tokenizer.encode_plus(
            example_batch,
            max_length=32,
            padding="max_length",
            add_special_tokens=True,
            pad_to_max_length=True,
            truncation=True,
        )
        encodings = {
            "input_ids": torch.tensor(encodings["input_ids"], device="cuda"),
            "attention_mask": torch.tensor(encodings["attention_mask"], device="cuda"),
        }
        return encodings

    @staticmethod
    def _generate_batch(example_batch: list) -> dict:
        from torch.nn.utils.rnn import pad_sequence

        input_ids = pad_sequence(
            [example["input_ids"] for example in example_batch], batch_first=True
        )
        attention_mask = pad_sequence(
            [example["attention_mask"] for example in example_batch], batch_first=True
        )
        batch = {"input_ids": input_ids, "attention_mask": attention_mask}
        return batch
