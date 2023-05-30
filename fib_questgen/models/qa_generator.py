import pickle
import random
import re
from typing import Dict, List, Optional, Tuple, Union

from fib_questgen.models.base_generator import FillInBlankGenerator
from fib_questgen.models.generators import (
    PhrasalVerbGenerator,
    VerbFormGenerator,
    WordFamilyGenerator,
)


class QAGenerator:
    def __init__(self):
        self.models = {
            "verb-form": VerbFormGenerator(),
            "word-family": WordFamilyGenerator(),
            "v-preposition": PhrasalVerbGenerator(),
        }

    @staticmethod
    def pick_random_context(domain: str = "english", level: str = "paragraph"):
        """
        Return a context in database randomly
        Args:
            level: "sentence" or "paragraph"
            domain: "english" or "history"
        Returns: A context in database

        """
        ENLISH_CONTEXT_DATA_PATH = "fib_questgen/data/english_context.pkl"
        data_map = {"english": ENLISH_CONTEXT_DATA_PATH}
        with open(data_map[domain], "rb") as f:
            data = pickle.load(f)
        result = random.choice(data)
        if level == "sentence":
            result = random.choice(result.split("."))
        return result

    @staticmethod
    def pick_index_pairs(question_list, answer_list):
        picked_index = []
        index_list = list(range(len(question_list)))
        random.shuffle(index_list)
        for index in index_list:
            if len(answer_list[index]) == 4:
                picked_index.append(index)
        return picked_index

    @staticmethod
    def normalize_paragraph(text):
        normal_punc = "!,.;:?"
        open_punc = "[({"
        close_punc = "])}"
        connect_punc = "-/_=+"

        for punc in normal_punc:
            text = text.replace(f" {punc} ", f"{punc} ")
            text = text.replace(f" {punc}", f"{punc}")
        for punc in open_punc:
            text = text.replace(f"{punc} ", punc)
        for punc in close_punc:
            text = text.replace(f" {punc}", punc)
        for punc in connect_punc:
            text = text.replace(f" {punc} ", punc)
        contractions = ["'m", "n't", "'s", "'ll ", "-", "'re", "'d", "'ve"]
        for ctrs in contractions:
            text = text.replace(f" {ctrs}", ctrs)
        text = re.sub(r"(?=\<)(.*?)(?<=\>)", "", text)
        return text.strip()

    def detach_quest(self, qa_pair) -> Dict[str, Union[List, str]]:
        word_list = qa_pair[0].split()
        for i, word in enumerate(word_list):
            if word == "___":
                quest = QAGenerator.expand_sentence(i, word_list)
                return {
                    "question": self.normalize_paragraph(quest),
                    "answers": qa_pair[1],
                }

    def generate_paragraph(self, num_blank: int = 2, context: str = ""):
        if not context:
            context = self.pick_random_context(domain="english", level="paragraph")
        question_list = []
        answer_list = []
        for model in self.models.values():
            qa_out = model.generate(context)
            question_list.extend(qa_out[0])
            answer_list.extend(qa_out[1])

        if num_blank < len(question_list):
            index_list = self.pick_index_pairs(question_list, answer_list)
            index_list = index_list[:num_blank]
        else:
            index_list = list(range(len(question_list)))

        qa_pairs = [(question_list[i], answer_list[i]) for i in index_list]
        qa_pairs = self.sort_result(qa_pairs)
        results = []
        for pair in qa_pairs:
            qa_item = self.detach_quest(pair)
            results.append(qa_item)
        return {"context": context, "results": results}

    def generate_sentence(
        self, context: str, word: Optional[Union[List, str]] = None
    ) -> Dict[str, List[Dict[str, List]]]:
        """
        Generate a fill-in-blank questions-answers from an English sentence.
        Args:
            context: An English sentence.
            word: Specific word to create a blank.

        Returns:
            Dict of context and result (fill-in-blank questions-answers)
        """
        results = list()
        question_list = list()
        answer_list = list()
        if not context:
            context = self.pick_random_context(domain="english", level="sentence")
        for model in self.models.values():
            quest, ans = self._generate(context=context, model=model, word=word)
            question_list.extend(quest), answer_list.extend(ans)
        index_list = list(range(len(question_list)))
        qa_pairs = [(question_list[i], answer_list[i]) for i in index_list]
        qa_pairs = self.sort_result(qa_pairs)
        for pair in qa_pairs:
            qa_item = self.detach_quest(pair)
            results.append(qa_item)
        return {"context": context, "results": results}

    @staticmethod
    def _generate(
        context: str,
        model: FillInBlankGenerator,
        word: Optional[Union[List, str]] = None,
    ) -> Tuple[List, List]:
        """
        Support using model to generate a FIB question-answering, with or without specific word.

        Args:
            context : An english paragraph or sentence.
            model : A Fill-In-Blank model.
            word : Word to turn into blank.

        Returns:
            Question list and Answer list.
        """
        question_list = list()
        answer_list = list()
        if isinstance(word, list):
            for item in word:
                qa_out = model.generate(context=context, word=item)
                question_list.extend(qa_out[0])
                answer_list.extend(qa_out[1])
        else:
            if isinstance(word, str):
                qa_out = model.generate(context=context, word=word)
            else:
                qa_out = model.generate(context=context)
            question_list.extend(qa_out[0])
            answer_list.extend(qa_out[1])
        return question_list, answer_list

    @staticmethod
    def check_blank(index, paragraph):
        check_range = (
            0 if index - 10 < 0 else index - 10,
            len(paragraph) - 1 if index + 10 > len(paragraph) - 1 else index + 10,
        )
        check_string = paragraph[check_range[0] : check_range[1]]
        for word in check_string.split():
            if word == "___":
                return False
        return True

    @staticmethod
    def expand_sentence(idx, word_list):
        sentence = [word_list[idx]]
        for i in range(idx + 1, len(word_list)):
            if "." in word_list[i] or "?" in word_list[i]:
                sentence.append(word_list[i])
                break
            sentence.append(word_list[i])
        for i in range(idx - 1, -1, -1):
            if "." in word_list[i] or "?" in word_list[i]:
                break
            sentence.insert(0, word_list[i])
        if sentence[-1] != ".":
            sentence.append(".")
        return " ".join(sentence)

    @staticmethod
    def indexing(paragraph, answers):
        index = 0
        qa_pair = []
        paragraph = paragraph.split()
        for i, w in enumerate(paragraph):
            if w == "___":
                paragraph[i] = f"__({index + 1})__"
                qa_pair.append(
                    {
                        "question": QAGenerator.expand_sentence(i, paragraph),
                        "answers": answers[index],
                    }
                )
                index += 1
        paragraph = " ".join(paragraph)
        return {
            "context": paragraph,
            "results": qa_pair,
        }

    def _merge_question(self, qa_pairs, num_blank_target):
        paragraph = qa_pairs[0][0]
        options = [qa_pairs[0][1]]
        num_blank = 1
        for i in range(1, len(qa_pairs)):
            question = qa_pairs[i][0]
            option = qa_pairs[i][1]
            index = question.split().index("___")
            if paragraph[index] != "___" and self.check_blank(index, paragraph):
                paragraph = paragraph.split()
                paragraph[index] = "___"
                option_index = paragraph[: index + 1].count("___")
                paragraph = " ".join(paragraph)
                options.insert(option_index - 1, option)
                num_blank += 1
                if num_blank == num_blank_target:
                    break
        return {"context": self.normalize_paragraph(paragraph), "answers": options}

    @staticmethod
    def sort_result(results):
        """Sort results by index of blanks index"""
        blank_idx = []
        for item in results:
            question = item[0]
            for i, word in enumerate(question.split()):
                if word == "___":
                    blank_idx.append(i)
        results = [x for _, x in sorted(zip(blank_idx, results))]
        return results
