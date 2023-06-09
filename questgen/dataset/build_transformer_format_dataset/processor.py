class DataProcessor:
    def __init__(
        self, tokenizer, model_type="t5", max_source_length=512, max_target_length=32
    ):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.model_type = model_type
        self.hl_token = "<hl>"

        if model_type == "t5":
            self.sep_token = "<sep>"
        elif model_type == "bart":
            self.sep_token = "<sep>"
        else:
            self.sep_token = "[SEP]"

    def process(self, dataset):
        if self.model_type == "t5":
            dataset = dataset.map(self._add_eos_examples)

        dataset = dataset.map(self._add_special_tokens)
        dataset = dataset.map(self._convert_to_features, batched=True)
        return dataset

    def _add_eos_examples(self, example):
        example["source_text"] = example["source_text"] + " </s>"
        example["target_text"] = example["target_text"] + " </s>"
        return example

    def _add_special_tokens(self, example):
        example["source_text"] = example["source_text"].replace(
            "{hl_token}", self.hl_token
        )
        example["source_text"] = example["source_text"].replace(
            "{sep_token}", self.sep_token
        )
        example["target_text"] = example["target_text"].replace(
            "{sep_token}", self.sep_token
        )
        return example

    # Tokenize the examples
    def _convert_to_features(self, example_batch):
        source_encoding = self.tokenizer.batch_encode_plus(
            example_batch["source_text"],
            max_length=self.max_source_length,
            padding="longest",
            pad_to_max_length=True,
            truncation=True,
        )

        target_encoding = self.tokenizer.batch_encode_plus(
            example_batch["target_text"],
            max_length=self.max_target_length,
            padding="longest",
            pad_to_max_length=True,
            truncation=True,
        )

        encodings = {
            "source_ids": source_encoding["input_ids"],
            "target_ids": target_encoding["input_ids"],
            "attention_mask": source_encoding["attention_mask"],
        }

        return encodings
