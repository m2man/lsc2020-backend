import json
from elasticsearch import Elasticsearch, exceptions
from tqdm import tqdm
import nltk
import re
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
import itertools as it

es = Elasticsearch([{"host": "localhost", "port": 9200}])

interest_index = "autosuggest"
# try:
#     es.indices.delete(index=interest_index)
# except exceptions.NotFoundError:
#     print("")


def is_location(word):
    word = word.split()[-1]
    for synset in wn.synsets(word):
        ss = synset
        while True:
            if len(ss.hypernyms()) > 0:
                ss = ss.hypernyms()[0]
                if ss in [wn.synset('structure.n.01'),
                          wn.synset('facility.n.01'),
                          wn.synset('organization.n.01'),
                          wn.synset('location.n.01'),
                          wn.synset('way.n.06')]:
                    return True
            else:
                break
    return False


def get_place_query(input_query):
    places = []
    text = word_tokenize(input_query)
    tags = nltk.pos_tag(text)
    for i, (word, tag) in enumerate(tags):
        if is_location(word) or tag == "NNP":
            j = i
            while j > 0:
                j -= 1
                if tags[j][1] not in ['NN', 'POS', 'NNP', 'JJ', 'DT', 'FW', 'JJR', 'JJS', 'NP', 'NPS', 'NNS']:
                    break
            places.append(' '.join(text[j + 1: i + 1]))

    place_query = " OR ".join(places)
    return place_query

#
# # Add analyzer for the client #
# es.indices.create(index=interest_index,
#                   body={"settings": {
#                       "analysis": {
#                           "analyzer": {
#                               "my_analyzer": {
#                                   "tokenizer": "standard",
#                                   "filter": ["classic", "lowercase", "trim"],
#                               }
#                           },
#                       }
#                   },
#                       "mappings": {
#                           "properties": {
#                               "suggest": {
#                                   "type": "completion",
#                                   "preserve_separators": False,
#                                   "analyzer": "my_analyzer",
#                                   "search_analyzer": "my_analyzer"},
#                               "extra": {
#                                   "type": "keyword",
#                               },
#                               "required_matches": {
#                                   "type": "long"
#                               }
#                           }
#                       }
#                   }
#                   )
gps_points = json.load(open('data/poi_scores.json'))
# poi = json.load(open('data/poi2.json'))
# poi_description = dict([(p["name"], p["full_location_description"]) for p in poi])
#
# for id, place in tqdm(enumerate(gps_points)):
#     official_name = place
#     variations = []
#     # official_name = place.split(',')[0].split('-')[0].split('/')[0].split('at')[0].strip('.')
#     official_name = re.sub(r'\([^)]*\)', '', official_name)
#     if len(official_name.split()) < 5:
#         shorten_name = word_tokenize(official_name.lower())
#         for i in range(1, len(shorten_name)):
#             variations.extend([" ".join(t).lower() for t in it.combinations(shorten_name, i + 1)])
#
#         tags = nltk.pos_tag(shorten_name)
#         for i, (word, tag) in enumerate(tags):
#             if tag == "NNP":
#                 variations = [word] + variations
#     else:
#         for i in range(len(official_name.split())):
#             variations.append(" ".join(official_name.split()[-i - 1:]))
#
#     # filter 1 word variations
#     document = {
#         "name": official_name,
#         "suggest": {
#             "input": variations,
#             "weight": gps_points[place]
#         },
#         "weight": gps_points[place],
#         "length": len(official_name),
#         "extra": list(set([p.lower() for p in poi_description[place]
#                            if len(p) > 3 and not any(char.isdigit() for char in p)])),
#         "required_matches": 2
#     }
#     res = es.index(index=interest_index, doc_type="_doc", id=id, body=document)

asigned_gps = json.load(open('data/u1_meta.json'))
addresses = set([asigned_gps[image]["address"] for image in asigned_gps])
for id, official_name in tqdm(enumerate(addresses), total=len(addresses)):
    if official_name == "":
        continue

    variations = []
    # official_name = place.split(',')[0].split('-')[0].split('/')[0].split('at')[0].strip('.')
    official_name = re.sub(r'\([^)]*\)', '', official_name)
    if len(official_name.split()) < 5:
        shorten_name = word_tokenize(official_name.lower())
        for i in range(1, len(shorten_name)):
            variations.extend([" ".join(t).lower() for t in it.combinations(shorten_name, i + 1)])

        tags = nltk.pos_tag(shorten_name)
        for i, (word, tag) in enumerate(tags):
            if tag == "NNP":
                variations = [word] + variations
    else:
        for i in range(len(official_name.split())):
            variations.append(" ".join(official_name.split()[-i - 1:]))

    # filter 1 word variations
    document = {
        "name": official_name,
        "suggest": {
            "input": variations,
            "weight": 2000
        },
        "weight": 2000,
        "length": len(official_name),
        "extra": "",
        "required_matches": 2
    }
    res = es.index(index=interest_index, doc_type="_doc", id=id + len(gps_points), body=document)

print("==========\nTotal number of document:")
res = es.search(index=interest_index, body={"query": {"match_all": {}}}, size=9999)  # Simple list all document
print(len(res["hits"]["hits"]))  # Number of result
