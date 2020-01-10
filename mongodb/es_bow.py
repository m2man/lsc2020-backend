'''
This is ES set up for bow embedding and text search for LSC
'''

from tqdm import tqdm
import elasticsearch
from elasticsearch import Elasticsearch
import json
from pathlib import Path
import pickle
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from datetime import datetime
import calendar
import random
import numpy as np
from nltk.tokenize import word_tokenize
import time

stop_words = stopwords.words('english')
stop_words += [',', '.']
ps = PorterStemmer()
Data_path = './data'

##### Load Synonym #####
Synonym_file_stemmed = Data_path + '/List_synonym_glove_all_stemmed.pickle'
with open(Synonym_file_stemmed, 'rb') as f:
    list_synonym_stemmed = pickle.load(f)

Synonym_file = Data_path + '/List_synonym_glove_all.pickle'
with open(Synonym_file, 'rb') as f:
    list_synonym = pickle.load(f)

##### Load description #####
description = json.load(open("data/u1_full_info.json"))

##### Load extended dictionary #### --> Also feature vector format
with open(Data_path + '/bow_my_dictionary.pickle', "rb") as f:
    my_dictionary = pickle.load(f)

numb_ft = len(my_dictionary)
print("Number of feature: " + str(numb_ft))

##### Load embedded bow for images #####
with open(Data_path + '/bow_feature_all.pickle', "rb") as f:
    bow_ft_images = pickle.load(f)

##### Load IDF #####
with open(Data_path + '/bow_idf.pickle', "rb") as f:
    my_idf = pickle.load(f)

##### Setting up ES #####
es = Elasticsearch([{"host": "localhost", "port": 9200}])

interest_index = "lsc2019_combined_text_bow"

print("Deleting index: " + interest_index)
try:
    es.indices.delete(index=interest_index)
except Exception:
    print("Do not have index to delete: " + interest_index)

########### Add analyzer for the client ##########
es.indices.create(
    index=interest_index,
    body={
        "settings": {
            # just one shard, no replicas for testing
            "number_of_shards": 1,
            "number_of_replicas": 0,

            # custom analyzer for analyzing file paths
            "analysis": {
                # Set up analyer --> include
                #   + Char_Filter(useless)
                #   + Tokenizer (token word)
                #   + Filter (tokenizer filter: filter for tokenized token) --> stem, synonym
                "analyzer": {
                    "analyzer_tfidf": {  # Set up analyzer first --> then define own custom thing later
                        "type": "custom",
                        "tokenizer": "tokenizer_tfidf",
                        "filter": [
                            "english_possessive_stemmer",
                            "lowercase",
                            "english_stop",
                            "english_keywords",
                            "english_stemmer"
                        ]
                    },
                    "analyzer_search": {
                        # Define another analyzer for search (usually the same with analyzer of index, but now we will do different)
                        "type": "custom",
                        "tokenizer": "standard",  # Just simply lower the query search
                        "filter": [
                            "my_graph_synonym",  # Synonym filter
                            "lowercase",
                            "english_stop",
                            "english_keywords",
                            "english_possessive_stemmer",
                            "english_stemmer"
                        ]
                    },
                    "analyzer_object_yolo_term": {
                        "type": "custom",
                        "tokenizer": "tokenizer_term",  # remove 'and' and ','
                        "filter": [
                            "lowercase",
                            "english_stop",
                            "english_keywords",
                            "english_possessive_stemmer",
                            "english_stemmer"
                        ]
                    },
                    "analyzer_object_yolo_term_clip": {
                        "type": "custom",
                        "tokenizer": "standard",  # remove 'and' and ','
                        "filter": [
                            "lowercase",
                            "english_stop",
                            "english_keywords",
                            "english_possessive_stemmer",
                            "english_stemmer"
                        ]
                    },
                    "analyzer_object_yolo_term_clip_search": {
                        "type": "custom",
                        "tokenizer": "standard",  # remove 'and' and ','
                        "filter": [
                            "my_graph_synonym",
                            "lowercase",
                            "english_stop",
                            "english_keywords",
                            "english_possessive_stemmer",
                            "english_stemmer"
                        ]
                    },
                    "analyzer_gps_description": {
                        "type": "custom",
                        "tokenizer": "standard",  # remove 'and' and ','
                        "filter": [
                            "address_shingle",
                            "lowercase",
                            "english_stop",
                            "english_possessive_stemmer",
                            "english_stemmer"
                        ]
                    },
                },
                "tokenizer": {
                    "tokenizer_tfidf": {
                        "type": "edge_ngram",
                        "min_gram": 1,
                        "max_gram": 10,
                        "token_chars": [
                            "letter",
                            "digit"
                        ]
                    },
                    "tokenizer_term": {
                        "type": "simple_pattern_split",
                        "pattern": "(( and )|(, ))"
                    }
                },
                "filter": {
                    "address_shingle": {
                            "type": "shingle",
                            "max_shingle_size": 3,
                            "output_unigrams": False,
                            "output_unigrams_if_no_shingles": False
                    },
                    "english_stop": {
                        "type": "stop",
                        "stopwords": "_english_"
                    },
                    "english_keywords": {
                        "type": "keyword_marker",
                        "keywords": ["example"]
                    },
                    "english_stemmer": {
                        "type": "stemmer",
                        "language": "english"
                    },
                    "english_possessive_stemmer": {
                        "type": "stemmer",
                        "language": "possessive_english"
                    },
                    # Should have synonym filter here
                    "my_synonym": {
                        "type": "synonym",
                        "synonyms_path": "analysis/all_synonym.txt"
                    },
                    "my_graph_synonym": {
                        "type": "synonym_graph",
                        "synonyms_path": "analysis/all_synonym.txt"
                    },
                    "edge_ngram_filter": {
                        "type": "edge_ngram",
                        "min_gram": 1,
                        "max_gram": 10
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "id":  {
                    "type": "keyword"
                },
                "scene": {
                    "type": "text",
                    "analyzer": "analyzer_tfidf",
                    "search_analyzer": "analyzer_search"
                },
                "description": {
                    "type": "text",
                    "analyzer": "analyzer_object_yolo_term",
                    "search_analyzer": "analyzer_object_yolo_term"
                },
                "description_clip": {
                    "type": "text",
                    "analyzer": "analyzer_object_yolo_term_clip",
                    "search_analyzer": "analyzer_object_yolo_term_clip_search"
                },
                "weekday": {
                    "type": "text",
                    "analyzer": "analyzer_gps_description",
                    "search_analyzer": "analyzer_gps_description"
                },
                "address": {
                    "type": "text",
                    "analyzer": "analyzer_gps_description",
                    "search_analyzer": "analyzer_gps_description"
                },
                "description_embedded": {
                    "type": "dense_vector",
                    "dims": numb_ft
                },
                "location": {
                    "type": "geo_point"
                },
                "time": {
                    "type": "date",
                    "format": "yyyy/MM/dd HH:mm:00+00"
                },
                "category_description1": {
                    "type": "keyword"
                },
                "category_description2": {
                    "type": "keyword"
                }
            }
        }
    }
)

# Get id images, both json and embedded file should have the same id list
list_id_images = list(description.keys())

for id, (image, desc) in tqdm(enumerate(description.items()), total=len(description)):
    if es.exists(interest_index, id):
        continue
    desc.update({"description_embedded": bow_ft_images[image].tolist(),
                 "location": {"lon": round(desc["location"][1], 5),
                              "lat": round(desc["location"][0], 5)} if desc["location"] else None})
    res = es.index(index=interest_index,
                   doc_type="_doc",
                   id=id,
                   body=desc)