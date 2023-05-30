from transformers import T5ForConditionalGeneration, T5Tokenizer

from graph.modules.triples_extract.incorporate import encode_with_graph


if __name__ == "__main__":
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    max_source_length = 128

    test_context = [
        "If training on TPU, it is recommended to pad all examples of the dataset to the same length or make use of pad_to_multiple_of to have a small number of predefined bucket sizes to fit all examples in.",
        "Dynamically padding batches to the longest example is not recommended on TPU as it triggers a recompilation for every batch shape that is encountered during training thus significantly slowing down the training.",
        "Only padding up to the longest example in a batch) leads to very slow training on TPU.",
    ]

    inputs = encode_with_graph(
        context=test_context,
        tokenizer=tokenizer,
        padding="max_length",
        max_length=max_source_length,
        truncation=True,
        return_tensors="pt",
    )

    out = model.generate(**inputs)
    print(out)
