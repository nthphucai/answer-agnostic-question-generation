import random
from os import PathLike
from typing import Any, List, Optional, Tuple

import nltk
from nltk.corpus import brown
from spacy.tokens import Doc, Token

from fib_questgen.models.base_generator import FillInBlankGenerator, MaskItem
from fib_questgen.utils import find_item_in_sublist
from fib_questgen.utils.constants import PREP_MAP


class VerbFormGenerator(FillInBlankGenerator):
    def __init__(self, vocab_name="verb_form.txt"):
        super().__init__(vocab_name)

    def _extract_mask(self, context: str, word: Optional[str] = None, **kwargs):
        doc = self.language_model(context)
        masks = []
        for i, token in enumerate(doc):
            if token.pos_ in ["VERB", "AUX"]:
                if word is None or token.text == word:
                    masks.append(MaskItem(word=token.text, index=i))
        word_list = [token.text for token in doc]
        return masks, word_list

    def _generate_options(self, answer: str, **kwargs):
        try:
            options = self.vocab[find_item_in_sublist(self.vocab, answer)]
            options.remove(answer)
            return options
        except TypeError:
            return None


class WordFamilyGenerator(FillInBlankGenerator):
    def __init__(self, vocab_name="word_family.txt"):
        super().__init__(vocab_name)

    def _extract_mask(self, context: str, word: Optional[str] = None, **kwargs):
        doc = self.language_model(context)
        masks = []
        for i, token in enumerate(doc):
            if word is None or token.text == word:
                masks.append(MaskItem(word=token.text, index=i))
        word_list = [token.text for token in doc]
        return masks, word_list

    def _generate_options(self, answer: str, **kwargs):
        try:
            options = self.vocab[find_item_in_sublist(self.vocab, answer)]
            options.remove(answer)
            return options
        except TypeError:
            return None


class PhrasalVerbGenerator(FillInBlankGenerator):
    def __init__(self):
        super().__init__(vocab_name=None)
        try:
            self.prep_choices = self._load_secondary_lm()
        except LookupError:
            nltk.download("brown")
            nltk.download("universal_tagset")
            self.prep_choices = self._load_secondary_lm()

    @staticmethod
    def _load_secondary_lm():
        return nltk.ConditionalFreqDist(
            (v[0], p[0])
            for (v, p) in nltk.bigrams(brown.tagged_words(tagset="universal"))
            if v[1] == "VERB" and p[1] == "ADP"
        )

    def _extract_mask(
        self, context: str, word: Optional[str] = None, **kwargs
    ) -> Tuple[List[MaskItem], List[str]]:
        doc = self.language_model(context)
        masks = list()
        verb_list = list()
        for i, token in enumerate(doc):
            masks, verb_list = self._extract_word(
                token=token,
                doc=doc,
                index=i,
                masks=masks,
                verb_list=verb_list,
                word=word,
            )
        word_list = [token.text for token in doc]
        self._set_context(key="verb", value=verb_list)
        return masks, word_list

    @staticmethod
    def _extract_word(
        token: Token,
        doc: Doc,
        index: int,
        masks: List[Any],
        verb_list: List[Any],
        word: Optional[str] = None,
    ):
        try:
            if token.pos_ == "VERB" and (
                doc[index + 1].pos_ == "PART" or doc[index + 1].pos_ == "ADP"
            ):
                if doc[index + 1].text == word or word is None:
                    masks.append(MaskItem(word=doc[index + 1].text, index=index + 1))
                    verb_list.append(token.text)
        except IndexError:
            pass
        return masks, verb_list

    def _generate_options(self, answer: str, **kwargs) -> Optional[List[str]]:
        verb = self.context["verb"].pop(0)
        options = [item[0] for item in self.prep_choices[verb].most_common(4)]
        try:
            options.remove(answer)
        except ValueError:
            pass
        return options


class PrepositionGenerator(FillInBlankGenerator):
    def __init__(self, vocab_name: PathLike = None):
        super().__init__(vocab_name)

    @staticmethod
    def normalize_punc(text: str) -> str:
        text = text.replace("$ ", "$")
        return text

    @staticmethod
    def _get_previous_token(current_idx: int, spacy_doc: Doc):
        if current_idx - 1 < 0:
            return "None"
        else:
            return spacy_doc[current_idx - 1]

    @staticmethod
    def check_consecutive(lst: List) -> bool:
        """"""
        return sorted(lst) == list(range(min(lst), max(lst) + 1))

    def _get_child_text(self, root_index: int, spacy_doc: Doc) -> str:
        """"""
        child_idx = self.get_all_child_idx(root_index, spacy_doc)
        child_idx.remove(root_index)
        if self.check_consecutive(child_idx):
            child_idx = sorted(child_idx)
            return " ".join([spacy_doc[i].text for i in child_idx])

    @staticmethod
    def get_all_child_idx(root_index: int, spacy_doc: Doc) -> List:
        child_idx = [root_index]
        child = spacy_doc[root_index].children
        if child:
            for c in child:
                child_idx.extend(PrepositionGenerator.get_all_child_idx(c.i, spacy_doc))
        return child_idx

    def detect_child(self, child: str, spacy_doc: Doc):
        for ent in spacy_doc.ents:
            try:
                if ent.text == self.normalize_punc(child):
                    return ent.label_
            except AttributeError:
                print("Warning...")
        return "ALL"

    def _extract_mask(
        self, context: str, word: Optional[str] = None, **kwargs
    ) -> Tuple[List[MaskItem], List[str]]:
        doc = self.language_model(context)
        masks = []
        label_list = []
        for token in doc:
            if word is None or token.text == word:
                if token.pos_ == "ADP":
                    child_text = self._get_child_text(token.i, doc)
                    label = self.detect_child(child_text, doc)
                    if label:
                        label_list.append(label)
                        masks.append(MaskItem(word=token.text, index=token.i))

        self._set_context(key="label", value=label_list)
        text_list = [token.text for token in doc]
        return masks, text_list

    def _generate_options(self, answer: str, **kwargs) -> Optional[List[str]]:
        dependency_label = self.context["label"].pop(0)
        answer_pool = PREP_MAP[dependency_label]
        options = random.choices(answer_pool, k=4)
        return options
