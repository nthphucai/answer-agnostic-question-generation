import os
from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fib_questgen.models.spacy_model import SpacyModelWrapper
from fib_questgen.utils import normalize_punc


@dataclass
class MaskItem:
    word: str
    index: int


class FillInBlankGenerator(ABC):
    def __init__(self, vocab_name, use_spacy=True):
        self.language_model = None
        self.vocab = None
        self.context = {}
        if vocab_name:
            self.vocab = self._load_vocab(vocab_name)
        if use_spacy:
            self.language_model = SpacyModelWrapper()

    @staticmethod
    def _load_vocab(vocab_name):
        module_path = Path(os.path.dirname(__file__)).parent
        vocab_path = os.path.join(module_path, "data", vocab_name)
        with open(vocab_path, "r") as f:
            data = f.readlines()
        data = [s.replace("\n", "").split() for s in data]
        return data

    def _set_context(self, key, value):
        self.context[key] = value

    @abstractmethod
    def _extract_mask(
        self, context: str, word: Optional[str] = None, **kwargs
    ) -> Tuple[List[MaskItem], List[str]]:
        """Generate blank for giving context."""
        pass

    @abstractmethod
    def _generate_options(self, answer: str, **kwargs) -> Optional[List[str]]:
        """Generate distractors from given correct answer."""
        pass

    @staticmethod
    def _is_list_of_words(args: Dict[str, Any]):
        """Check keyword argument 'word'."""
        if "word" in args:
            if isinstance(args["word"], list):
                return True
            elif isinstance(args["word"], str):
                return False
        else:
            return None

    def _initialize_extract_mask(self, context: str, **kwargs) -> Tuple[List, List]:
        """
        Create blank point. If specific 'word' was passed, only create blank point with that word.
        Args:
            context (str): Input context to create blank point
            **kwargs: Additional specific kwargs (supporting `word`)

        Returns: A Tuple with the following items:

            - masks (`List[MaskItem]`): List of word was create blank
            - words (`List[str]`): List of origin words
        """
        _is_list_of_words = self._is_list_of_words(kwargs)
        if _is_list_of_words is not None:
            words = kwargs["word"]
            import logging

            if not _is_list_of_words:
                logging.warning(f"kwargs['word'] {words}")
                return self._extract_mask(context=context, word=words)
            else:
                masks, word_list = list(), list()
                for word in words:
                    child_masks, word_list = self._extract_mask(
                        context=context, word=word
                    )
                    masks.extend(child_masks)
                return masks, word_list
        else:
            return self._extract_mask(context=context)

    @staticmethod
    def _create_blank(word_list: List[str], index: int) -> str:
        """Create blank for given context."""
        word_list[index] = "___"
        return normalize_punc(" ".join(word_list))

    def generate(self, context, **kwargs) -> Tuple[List, List[List[str]]]:
        """Generate fill-in-blank questions and answers."""
        masks, word_list = self._initialize_extract_mask(context=context, **kwargs)
        questions = []
        answers = []
        for mask_item in masks:
            mask_word = mask_item.word
            mask_index = mask_item.index
            options = self._generate_options(answer=mask_word, **kwargs)
            if options is not None:
                question = self._create_blank(copy(word_list), mask_index)
                options.insert(0, mask_word)
                answers.append(options[:4])
                questions.append(question)
        return questions, answers
