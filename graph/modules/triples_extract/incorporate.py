from typing import List

import torch
from transformers import PreTrainedTokenizer

from graph.modules.triples_extract.extract_triples import TripleExtractor


def encode_with_graph(context: List[str], tokenizer: PreTrainedTokenizer, **kwargs):
    """
    Concat graph vector and context vector
    Args:
        context: list of context
        tokenizer: pretrained tokenizer
        **kwargs: encode args

    Returns:

    """
    graph_list = [TripleExtractor(text) for text in context]
    context_encoding = tokenizer(context, **kwargs)
    graph_encoding = tokenizer(
        [graph.get_string_presentation() for graph in graph_list], **kwargs
    )
    inputs = {
        "input_ids": torch.cat(
            (graph_encoding["input_ids"], context_encoding["input_ids"]), dim=1
        ),
        "attention_mask": torch.cat(
            (graph_encoding["attention_mask"], context_encoding["attention_mask"]),
            dim=1,
        ),
    }
    return inputs
