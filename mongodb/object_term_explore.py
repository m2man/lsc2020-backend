import json
import pickle
from collections import Counter
from nltk import word_tokenize
from nltk.stem import PorterStemmer

# Load Synonym
Synonym_file_stemmed = 'data/List_synonym_glove_all_stemmed.pickle'
with open(Synonym_file_stemmed, 'rb') as f:
    list_synonym_stemmed = pickle.load(f)

Synonym_file = 'data/List_synonym_glove_all.pickle'
with open(Synonym_file, 'rb') as f:
    list_synonym = pickle.load(f)

# # Load description
# object_description = json.load(open('data/u1_description.json'))
# words = word_tokenize(", ".join([", ".join([desc["scene_image"], desc["object_image"]]) for key, desc in object_description.items()]))
# all_words = [(word, count) for (word, count) in Counter(words) if len(word) > 2])

##### Load dictionary #### --> Also feature vector format
with open('data/bow_my_dictionary.pickle', "rb") as f:
    my_dictionary = pickle.load(f)
print(my_dictionary)
def find_synonym(word, dictionary=my_dictionary, list_synonym=list_synonym_stemmed):
    # word need to be stemmed first
    result = []
    if word in dictionary:
        return [word]
    else:
        index_row = [word in list_synonym[i] for i in range(len(list_synonym))]
        try:
            index_row = index_row.index(True)
            result = [list_synonym[index_row][0]]
            return result
        except ValueError:
            return result

ps = PorterStemmer()
for word in word_tokenize("parking my car at the parking lot and eat an apple given by a women"):
    word = word.lower()
    print(word, find_synonym(ps.stem(word)))