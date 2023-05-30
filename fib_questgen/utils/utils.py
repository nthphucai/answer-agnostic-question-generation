def normalize_punc(text):
    punc = ["!:.,?$%;"]
    for p in punc:
        text = text.replace(" " + p, p)
    return text


def find_item_in_sublist(parent_list, item):
    for x, child_list in enumerate(parent_list):
        if item in child_list:
            return x
