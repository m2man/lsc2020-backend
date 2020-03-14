import time
import json
import os
import shelve
from collections import defaultdict

from gensim.models import Word2Vec
from nltk import bigrams
from nltk.corpus import wordnet as wn
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize, MWETokenizer
try:
    from pattern.en import lemma, singularize
except:
    from pattern3.en import lemma, singularize


def shelve_it(file_name):
    d = shelve.open(file_name)

    def decorator(func):
        def new_func(param):
            if param not in d:
                d[param] = func(param)
            return d[param]

        return new_func

    return decorator


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' %
                  (method.__name__, (te - ts) * 1000))
        return result

    return timed


def get_most_similar(model, word, vocabulary):
    if word in model.wv.vocab:
        vocabulary = [w for w in vocabulary if w in model.wv.vocab]
        if vocabulary:
            return sorted(zip(vocabulary, model.wv.distances(word, vocabulary)), key=lambda x: x[1])
    else:
        print("Word not in word2vec model")
    return []


def hyper(s): return s.hypernyms()


def hypo(s): return s.hyponyms()


def holo(s): return s.member_holonyms() + \
    s.substance_holonyms() + s.part_holonyms()


def mero(s): return s.member_meronyms() + \
    s.substance_meronyms() + s.part_meronyms()


def list_lemmas(synsets_list):
    for synset in synsets_list:
        for lemma in synset.lemmas():
            yield lemma.name()


def update_lemmas(lemma_dicts, synsets_list, depth):
    for lemma in list_lemmas(synsets_list):
        if lemma not in lemma_dicts:
            lemma_dicts[lemma] = depth
        else:
            lemma_dicts[lemma] = min(lemma_dicts[lemma], depth)
    return lemma_dicts


@shelve_it('hypernyms.shelve')
def get_hypernyms(word):
    syns = wn.synsets(word, pos='n')
    results = {}
    for syn in syns:
        for depth in range(10):
            if depth == 0:
                for lemma in syn.lemmas():
                    results[lemma.name()] = 0
            else:
                results = update_lemmas(
                    results, syn.closure(hyper, depth=depth), depth)
    return results


@shelve_it('keyword_cache.shelve')
def inspect(params):
    word, max_depth = params.split('/')
    max_depth = int(max_depth)
    syns = wn.synsets(word, pos='n')
    result = {"lemmas": {}, "hypernyms": {},
              "hyponyms": {}, "holonyms": {}, "meronyms": {}}
    for syn in syns:
        for lemma in syn.lemmas():
            result["lemmas"][lemma.name()] = 0
        for depth in range(max_depth):
            result["hypernyms"] = update_lemmas(
                result["hypernyms"], syn.closure(hyper, depth=depth), depth)
            result["hyponyms"] = update_lemmas(
                result["hyponyms"], syn.closure(hypo, depth=depth), depth)
            result["holonyms"] = update_lemmas(
                result["holonyms"], syn.closure(holo, depth=depth), depth)
            result["meronyms"] = update_lemmas(
                result["meronyms"], syn.closure(mero, depth=depth), depth)
    return result


# @shelve_it('similar_cache.shelve')
def get_similar(word):
    kws = [kw.keyword for kw in KEYWORDS]
    if word in kws:
        return [word]
    similars = defaultdict(lambda: 10)

    for kw in KEYWORDS:
        is_sim, new_depth = kw.is_similar(word)
        if is_sim:
            similars[kw.keyword] = min(new_depth, similars[kw.keyword])
    return list(similars.keys())


class Keyword:
    def __init__(self, word):
        word = word.replace(' ', '_')
        if word in ["waiting_in_line", "using_tools"]:
            self.keyword = word
        else:
            if word == "bakery/shop":
                self.keyword = "bakery"
            else:
                self.keyword = word
            depth = 0 if self.keyword in ["animal", "person", "food"] else 3
            self.words = inspect(f"{self.keyword}/{depth}")

    def is_similar(self, word):
        word = word.replace(' ', '_')
        if word == self.keyword:
            return True, -1
        if self.keyword in ["waiting_in_line", "using_tools"] or self.words is None:
            return False, 10
        if self.words:
            if word in self.words["lemmas"]:
                return True, 0

        for nyms in self.words:
            if word in self.words[nyms]:
                return True, self.words[nyms][word]

        hypernyms = get_hypernyms(word)
        if self.keyword in hypernyms:
            return True, hypernyms[self.keyword]
        else:
            return False, 10


specials = {"cloudy": "cloud"}


@shelve_it('process_cache.shelve')
def process_word(word):
    words = word.replace('_', ' ').replace('/', ' ').split()
    new_words = []
    for word, tag in pos_tag(words):
        if word in specials:
            new_words.append(specials[word])
        else:
            if tag == "NNS":
                new_words.append(singularize(word))
            else:
                new_words.append(lemma(word))
    return new_words


COMMON_PATH = os.getenv("COMMON_PATH")
vocabulary = json.load(open(f'{COMMON_PATH}/all_keywords.json'))
KEYWORDS = [Keyword(kw) for kw in vocabulary]
model = Word2Vec.load(f"{COMMON_PATH}/word2vec.model")
map2deeplab = json.load(open(f"{COMMON_PATH}/map2deeplab.json"))
deeplab2simple = json.load(open(f"{COMMON_PATH}/deeplab2simple.json"))
simples = json.load(open(f"{COMMON_PATH}/simples.json"))

tokenizer = MWETokenizer()
for kw in vocabulary:
    tokenizer.add_mwe(kw.split('_'))


def to_deeplab(word):
    for kw in map2deeplab:
        if word in map2deeplab[kw][1]:
            yield deeplab2simple[kw]


def get_all_similar(words, must_not_terms):
    shoulds = set()
    musts = set()
    for word in words:
        print("-" * 10)
        print(word)
        for w in to_deeplab(word):
            if w in vocabulary:
                musts.add(w)
        for w in process_word(word):
            similars = get_similar(w)
            if similars:
                if w in similars:
                    musts.add(w)
                    shoulds.update(similars)
                else:
                    shoulds.update(similars)
            for w2, dist in get_most_similar(model, w, vocabulary)[:20]:
                if dist < 0.1:
                    musts.add(w2)
                elif dist < 0.26:
                    shoulds.add(w2)
                print(w2.ljust(20), round(dist, 2))

    musts = musts.difference(must_not_terms)
    musts = musts.difference(["airplane", "plane"])
    print(musts, shoulds)
    shoulds = shoulds.difference(must_not_terms)
    return list(musts), list(shoulds)


def process_string(string, must_not_terms):
    tokens = tokenizer.tokenize(word_tokenize(string.lower()))
    pos = pos_tag(tokens)
    s = []
    for w, t in pos:
        if w == "be":
            continue
        if t in ["NN", "VB"]:
            s.append(w)
        elif t in ["NNS", "VBG", "VBP", "VBZ", "VBD", "VBN"]:
            w = singularize(lemma(w))
            if w != "be":
                s.append(w)
    return get_all_similar(tokenizer.tokenize(s), must_not_terms)


if __name__ == "__main__":
    # get_most_similar_to_multiple(word2vec, ["clock", "flower", "lamp", "monster", "rabbit", "bed"])
    print(process_string(
        "Eating fishcakes, bread and salad after preparing my presentation in powerpoint. It must have been lunch time. There was a guy in a blue sweater. I think there were phones on the table. After lunch, I made a coffee."))
