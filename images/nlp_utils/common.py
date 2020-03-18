import json
import re

import nltk
from nltk.corpus import stopwords
import os

stop_words = stopwords.words('english')
stop_words += [',', '.']

COMMON_PATH = os.getenv("COMMON_PATH")
simpletime = ['at', 'around', 'about', 'on']
period = ['while', "along", "as"]

preceeding = ['before', "afore"]
following = ['after']
location = ['across', 'along', 'around', 'at', 'behind', 'beside', 'near', 'by', 'nearby', 'close to',
            'next to', 'from', 'in front of', 'inside', 'in', 'into', 'off', 'on',
            'opposite', 'out of', 'outside', 'past', 'through', 'to', 'towards']

all_words = period + preceeding + following
all_prep = simpletime + period + preceeding + following
pattern = re.compile(f"\s?({'|'.join(all_words)}+)\s")

grouped_info_dict = json.load(open(f"{COMMON_PATH}/grouped_info_dict.json"))
locations = set([img["location"].lower()
                 for img in grouped_info_dict.values()])
regions = set([w.strip().lower() for img in grouped_info_dict.values()
               for w in img["region"]])
keywords = set([w.replace('_', ' ') for img in grouped_info_dict.values()
                for w in img["deeplab_concepts"] + img["concepts"] + img["attributes"] + img["category"]])

json.dump(list(keywords), open(f'{COMMON_PATH}/all_keywords.json', 'w'))
all_address = '|'.join([re.escape(a) for a in locations])
activities = set(["walking", "airplane", "transport"])
phrases = json.load(open(f'{COMMON_PATH}/phrases.json'))


def find_regex(regex, text, escape=False):
    regex = re.compile(regex, re.IGNORECASE + re.VERBOSE)
    for m in regex.finditer(text):
        result = m.group()
        start = m.start()
        while len(result) > 0 and result[0] == ' ':
            result = result[1:]
            start += 1
        while len(result) > 0 and result[-1] == ' ':
            result = result[:-1]
        yield (start, start + len(result), result)


def flatten_tree(t):
    return " ".join([l[0] for l in t.leaves()])


def flatten_tree_tags(t, pos):
    if isinstance(t, nltk.tree.Tree):
        if t.label() in pos:
            return [flatten_tree(t), t.label()]
        else:
            return [flatten_tree_tags(l, pos) for l in t]
    else:
        return t
