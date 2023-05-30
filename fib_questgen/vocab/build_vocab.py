import pickle
from typing import List

import spacy
from tqdm import tqdm


def remove_uppercase_word(word_list: List[str]):
    word_list = map(lambda x: x.lower(), word_list)
    word_list = list(dict.fromkeys(word_list))
    return word_list


def get_lemma_dictionary(vocab: List[str]):
    model = spacy.load("en_core_web_sm")
    dictionary = {}
    for word in tqdm(vocab):
        doc = model(word)
        if doc[0].lemma_ not in dictionary:
            dictionary[doc[0].lemma_] = [word]
        else:
            dictionary[doc[0].lemma_].append(word)
    return dictionary


if __name__ == "__main__":
    with open("data/english_vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    dictionary = get_lemma_dictionary(vocab)
    print(len(dictionary))
