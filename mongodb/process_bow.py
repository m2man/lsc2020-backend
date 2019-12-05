import json
import pickle
import numpy as np
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
Folder_Path = './'
stop_words = stopwords.words('english')
stop_words += [',', '.']
ps = PorterStemmer()


# Process stemmed synonym
File = Folder_Path + 'data/List_synonym_glove_all.pickle'
with open(File, 'rb') as f:
    list_synonym = pickle.load(f)
list_synonym_stemmed = []
for i in range(len(list_synonym)):
    synonym = list_synonym[i]
    stemmed = [ps.stem(word) for word in synonym]
    list_synonym_stemmed.append(stemmed)
File = Folder_Path + 'data/List_synonym_glove_all_stemmed.pickle'
with open(File, 'wb') as f:
    pickle.dump(list_synonym_stemmed, f)
'''

# Create dictionary vector
'''
File = Folder_Path + 'data/u1_full_info.json'
with open(File) as json_file:
    combined_description = json.load(json_file)
dictionary = []
for id_image, content_image in combined_description.items():
    tokens = word_tokenize(content_image["description"])
    stemmed_tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    dictionary += stemmed_tokens
    dictionary = sorted(list(set(dictionary)))
print('Total number of feature: {}'.format(len(dictionary)))
File = Folder_Path + 'data/bow_my_dictionary.pickle'
with open(File, 'wb') as f:
    pickle.dump(dictionary, f)
'''

# Create term frequency
'''
File = Folder_Path + 'data/bow_my_dictionary.pickle'
with open(File, 'rb') as f:
    my_dictionary = pickle.load(f)
all_tf_count = np.zeros(len(my_dictionary))
all_tf_dict = {}
for id_image, content_image in combined_description.items():
    tf_image = np.zeros(len(my_dictionary))
    tokens = word_tokenize(content_image["description"])
    stemmed_tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    for word in stemmed_tokens:
        try:
            index_tf = my_dictionary.index(word)
            tf_image[index_tf] += 1
        except Exception:
            print("File {} have exotic word {}". format(id_image, word))
    all_tf_count += tf_image
    if np.max(tf_image) > 0:
        tf_image = tf_image / np.sqrt(np.sum(tf_image ** 2))
    all_tf_dict[id_image] = tf_image
File = Folder_Path + 'data/bow_tf_all.pickle'
with open(File, 'wb') as f:
    pickle.dump(all_tf_dict, f)
File = Folder_Path + 'data/bow_tf_count.pickle'
with open(File, 'wb') as f:
    pickle.dump(all_tf_count, f)
'''

# Create IDF
'''
File = Folder_Path + 'data/bow_tf_count.pickle'
with open(File, 'rb') as f:
    all_tf_count = pickle.load(f)
number_of_document = len(combined_description)
my_idf = np.zeros(len(all_tf_count))
my_idf = np.log(number_of_document/all_tf_count)
File = Folder_Path + 'data/bow_idf.pickle'
with open(File, 'wb') as f:
    pickle.dump(my_idf, f)

# Create Bow Feature for each images
File = Folder_Path + 'data/bow_tf_all.pickle'
with open(File, 'rb') as f:
    all_tf_dict = pickle.load(f)

File = Folder_Path + 'data/bow_idf.pickle'
with open(File, 'rb') as f:
    my_idf = pickle.load(f)

bow_feature_all = {}
for id_image, tf_image in all_tf_dict.items():
    bow_feature_image = tf_image * my_idf
    if np.max(bow_feature_image) > 0:
        bow_feature_image /= np.sqrt(np.sum(bow_feature_image ** 2))
    bow_feature_all[id_image] = bow_feature_image

File = Folder_Path + 'data/bow_feature_all.pickle'
with open(File, 'wb') as f:
    pickle.dump(bow_feature_all, f)