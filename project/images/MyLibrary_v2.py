import json
import numpy as np
from operator import itemgetter
from collections import Counter
import pickle
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk import pos_tag
import random
import time
from datetime import datetime
import parsedatetime as pdt  # $ pip install parsedatetime
import spacy

nlp = spacy.load("en_core_web_sm")

c = pdt.Constants()
c.uses24 = True

cal = pdt.Calendar(c)


def get_time_query(text):
    doc = nlp(text)
    query = ["doc['time'].size()!=0"]

    for ent in doc.ents:
        day_available = False
        time_available = False
        at = text[ent.start_char - 3: ent.start_char - 1] == "at"
        before = text[ent.start_char - 7: ent.start_char - 1] == "before"
        after = text[ent.start_char - 6: ent.start_char - 1] == "after"

        if ent.label_ == "TIME":
            time_available = True
            time = (ent.text, before, after)
        elif (at or before or after) and ent.label_ == "CARDINAL":
            i = text.lower().find("am", ent.start_char)
            if i == -1:
                i = text.lower().find("pm", ent.start_char)
            if i != -1:
                time_available = True
                time = (text[ent.start_char: i + 2], before, after)

        if time_available:
            parsed = cal.parseDT(time[0])[0]
            if time[1]:
                query.append(f"doc['time'].value.getHour() <= {parsed.hour}")
            elif time[2]:
                query.append(f"doc['time'].value.getHour() >= {parsed.hour}")
            else:
                query.append(f"Math.abs(doc['time'].value.getHour() - {parsed.hour}) < 0.5")

        if ent.label_ == "DATE":
            day_available = True
            day = (ent.text, before, after)

        if day_available:
            parsed = cal.parseDT(day[0])[0]
            if day[1]:
                query.append(f"doc['time'].value.getDayOfMonth() <= {parsed.day}")
            elif day[2]:
                query.append(f"doc['time'].value.getDayOfMonth() >= {parsed.day}")
            else:
                query.append(f"doc['time'].value.getDayOfMonth() == {parsed.day}")

    return " && ".join(query)


stop_words = stopwords.words('english')
stop_words += [',', '.']
ps = PorterStemmer()

Datapath = "/Users/allietran/PycharmProjects/LSC/django/mongodb/data"

###### Load Synonym ########
Synonym_glove_all_file = Datapath + "/List_synonym_glove_all.pickle"
with open(Synonym_glove_all_file, "rb") as f:
    list_synonym = pickle.load(f)

##### Load Synonym stemmed #####
Synonym_file = Datapath + '/List_synonym_glove_all_stemmed.pickle'
with open(Synonym_file, 'rb') as f:
    list_synonym_stemmed = pickle.load(f)

##### Load dictionary #### --> Also feature vector format
with open(Datapath + '/bow_my_dictionary.pickle', "rb") as f:
    my_dictionary = pickle.load(f)

##### Load embedded bow for images #####
with open(Datapath + '/bow_feature_all.pickle', "rb") as f:
    bow_ft_images = pickle.load(f)

##### Load IDF #####
with open(Datapath + '/bow_idf.pickle', "rb") as f:
    my_idf = pickle.load(f)


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


def create_bow_ft_sentence(sentence, dictionary=my_dictionary, list_synonym=list_synonym_stemmed, idf=my_idf):
    word_tokens = word_tokenize(sentence.lower())
    good_tokens = [word for word in word_tokens if word not in stop_words]
    remove_stop_sentence = ' '.join(good_tokens)
    refined_words = [ps.stem(word) for word in good_tokens]
    # refined_words = sorted(list(set(refined_words)))
    term_freq = np.zeros(len(dictionary))
    for word in refined_words:
        word_synonym = find_synonym(word, dictionary, list_synonym)
        if len(word_synonym) == 0:
            continue
        else:
            word = word_synonym[0]
            if word in dictionary:
                word_index = dictionary.index(word)
            else:
                word_index = -1
            term_freq[word_index] += 1
    if np.max(term_freq) > 0:
        term_freq /= np.sqrt(np.sum(term_freq ** 2))  # Tu tf version
    bow_ft = term_freq * idf
    if np.max(bow_ft) > 0:
        bow_ft /= np.sqrt(np.sum(bow_ft ** 2))
    return bow_ft, remove_stop_sentence


def unlist(l):
    '''
    Unlist list of lists l into 1 list
    l = [[1, 2], [3]] --> result = [1, 2, 3]
    '''
    result = []
    for s in l:
        result.extend(s)
    return result


def search_es(ES, index, request, percent_thres=0.9, max_len=10):
    '''
    ES: Elasticsearch engine with collected data
    index: name of the index in ES
    reques: request in json format
    percent_thers: percentage of maxscore to be a threshold to filt the result
    max_len: maximum result that you want to get
    Output
    --> res: original result from elasticsearch
    --> result: list of id_image
    '''
    result = None

    res = ES.search(index=index, body=request, size=9999)  # Show all result (9999 results if possible)
    numb_result = len(res["hits"]["hits"])
    print("Total searched result: " + str(numb_result))

    if numb_result != 0:
        score = [d["_score"] for d in res["hits"]["hits"]]  # Score of all result (higher mean better)
        id_result = [d["_source"]["id"] for d in res["hits"]["hits"]]  # List all id images in the result

        score_np = np.asarray(score)

        max_score = score_np[0]
        thres_score = max_score * percent_thres

        index_filter = np.where(score_np > thres_score)[0]

        if len(index_filter) > 1:
            result = list(itemgetter(*list(index_filter))(id_result))
            numb_result = len(result)
        else:
            result = id_result[0]
            numb_result = 1

        if numb_result > max_len:
            result = result[0:max_len]

        print("Total remaining result: " + str(numb_result))  # Number of result
        print("Total output result: " + str(min(max_len, numb_result)))  # Number of result

    return res, result


def generate_original_synonym(given_word, list_synonym=list_synonym):
    '''
    Generate the synonym of given_word and these synonym are the name of the classes in yolo/cbnet
    list_synonym: is the list of synonym generated by Glove.iypn from colab, then download
    Output:
        + result: list of original synonym word of given_word
        + If can find any synonym --> len = 0
    '''
    result = []
    for index, sublist in enumerate(list_synonym):
        if given_word in sublist:
            result.append(list_synonym[index][0])
    return result


def generate_subterm_query(q, list_synonym=list_synonym):
    '''
    divided query q into subterm then find original synonym term for each subterm, and return the result
    list_synonym: see generate_original_synonym
    For ex: q = "1 car, 2 person" --> result = [[1 car, 1 truck, 1 motorbike], [2 person, 2 people]]
    Also give the original term --> original_result = [[1 car], [2 person]] # No synonym
    '''
    result = []
    q_process = q.replace(" and ", ", ")  # replace " and " to ", " --> "1 car and 1 person" to "1 car, 1 person"
    q_process = q_process.replace(" or ", ", ")  # same with "or" --> we dont care difference between "and" and "or"
    q_process = q_process.replace(", ", ",")  # "1 car, 1 person" to "1 car,1 person"
    q_split = q_process.split(",")  # "1 car" "1 person"
    original_result = [[x] for x in q_split if x != '']
    for tag in q_split:
        element = tag.split(" ")  # "1 car" --> "1", "car"
        try:  # Split number and object
            element_numb = int(element[0])
            element_object = " ".join(element[1:])
        except ValueError:
            element_numb = 0
            element_object = " ".join(element)
        original_synonym = generate_original_synonym(element_object, list_synonym)  # Find the original synonym
        if len(original_synonym) == 0:
            original_synonym = [element_object]
        if element_numb == 0:
            result.append([x for x in original_synonym])
        else:
            result.append([str(element_numb) + " " + x for x in original_synonym])
    return original_result, result


def generate_querystringquery_and_subquery(sq, max_change=1):
    '''
    Generate query string format from the subterm query sq (generated from generate_subterm_query)
    Also generate subquery from the subterm query sq
    max_change >= 0 --> maximum change value for number of object (below and above the original value)
    For Ex: sq = [[1 car, 1 truck, 1 motorbike], [2 person]], max_change = 1
    query_string_full = "("1 car" OR "1 truck" OR "1 motorbike") AND ("2 person" OR "2 people")"
    query_string_term = [[("1 car" OR "1 truck" OR "1 motorbike"), ("2 person" OR "2 people")]] --> subterm of query_string_full
    query_string_term_adjust = [[("2 car" OR "2 truck" OR "2 motorbike")], [ ... ]] --> adjust quantity with max_change
    '''
    query_string_full = ""
    query_string_term_adjust = []
    query_string_term = []
    change_seq = np.linspace(-max_change, max_change, 2 * max_change + 1)
    change_seq = np.delete(change_seq, max_change)  # remove 0 out of vector --> we dont keep original term
    for term in sq:
        query_string = "("
        for subterm in term:
            query_string += "(" + subterm + ")" + " OR "
        query_string = query_string[0:-4] + ")"
        query_string_term.append([query_string])
        query_string_full += query_string + " AND "
    for term in query_string_term:
        assert (len(term) == 1)
        term_str = term[0]
        term_str = term_str.replace("(", "")
        quantity = term_str.split()
        try:
            quantity = int(quantity[0])
            quantity_change = quantity + change_seq
            quantity_change = np.delete(quantity_change,
                                        np.where(quantity_change <= 0)[0])  # remove negative and 0 number
        except ValueError:
            quantity_change = np.array([])
        if (len(quantity_change) > 0):
            for i in quantity_change:
                term_change = term[0].replace(str(quantity), str(int(i)))
                query_string_term_adjust.append([term_change])
    query_string_full = query_string_full[0:-5]
    return query_string_full, query_string_term, query_string_term_adjust


def generate_query_embedding(sentence, numb_get_result=100):
    embedded_query, _ = create_bow_ft_sentence(sentence, my_dictionary, list_synonym_stemmed, my_idf)
    embedded_query = embedded_query.tolist()

    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_embedded, doc['description_embedded']) + 1.0",
                "params": {"query_embedded": embedded_query}
            }
        }
    }

    body = {
        "size": numb_get_result,
        "query": script_query,
        "_source": {"includes": ["id", "description", "time", "location", "address", "nearby POI", "driving", "weekday"]
                    }
    }

    body = json.dumps(body)
    return body


def create_json_query_string_part(query, field, boost=1):
    result = {
        "query_string": {
            "query": query,
            "default_field": field,
            "boost": boost
        }
    }
    return result


import textdistance

address_and_gps = json.load(open(Datapath + "/address_and_gps.json"))


def create_location_query(sentence, field, boost=1000):
    location_part = get_places(sentence)  # Put the location detected here
    location_decay = []
    new_location_part = []
    if len(location_part) > 0:
        for location in location_part:
            print(location)
            print("dcu car park 1" in address_and_gps)
            min_dist = 0.3
            min_location = None
            for address in address_and_gps:
                dist = textdistance.sorensen.distance(location.split(), address.split())
                if dist < min_dist and not address_and_gps[address][0] is None:
                    min_dist = dist
                    min_location = address
                if dist == min_dist and not address_and_gps[address][0] is None and \
                        abs(len(address) - len(location)) < abs(len(min_location) - len(location)):
                    min_dist = dist
                    min_location = address

            if min_dist < 0.3 and not min_location is None:
                location_decay.append({
                    "function_score": {
                        "gauss": {
                            "location": {
                                "origin": {"lat": address_and_gps[min_location][0][0],
                                           "lon": address_and_gps[min_location][0][1]},
                                "scale": f"{address_and_gps[min_location][1]}km",
                                "decay": 0.5
                            }
                        },
                        "boost": 50
                    }})
                location = min_location
                sentence = sentence.replace(location, "").replace("  ", " ")

            new_location_part.append(location)
            print(new_location_part)
        location_part = " OR ".join(new_location_part)
        stt = True
        location_json = create_json_query_string_part(query=location_part, field=field, boost=boost)
        print(sentence, '|', new_location_part)

    else:
        stt = False
        location_json = 0

    return sentence, stt, location_json, location_decay


def generate_query_combined(q, max_change=1, tie_breaker=0.7, numb_of_result=100):
    '''
    Quite the same with generate_es_query_dismax but now inclide query string query, not multimatch anymore
    Generate elastic-formatted request and use the result for the input of elasticsearch
    list_synonym: list of synonym generated from Glove and only support classes in yolo or cbnet or your own defined
    max_change >= 0: See generate_near_query
    tie_breaker: See elasticsearch document
    Output:
        + request_string is the txt format of the elasticsearch formatted request
    '''
    q = q.lower()
    having_comma = q.find(",")
    if having_comma > 0:
        having_comma = True
    else:
        having_comma = False
    result = "{\"size\":" + str(numb_of_result)
    result += ",\"_source\": {\"includes\": [\"id\", \"description\", \"time\", \"location\", \"address\", \"nearby POI\", \"driving\", \"weekday\" ]}"
    result += ",\"query\": {\"bool\": {\"must\": {\"dis_max\":{\"queries\":["
    queries_part = []
    q, having_location, location_query, location_decay = create_location_query(q, field="address", boost=10)
    embedded_query, adjust_sentence_query = create_bow_ft_sentence(q, my_dictionary, list_synonym_stemmed, my_idf)
    embedded_query = embedded_query.tolist()
    queries_part += [create_json_query_string_part(query=adjust_sentence_query, field="description_clip", boost=5)]
    if having_location:
        queries_part += [location_query]
    if having_comma:  # Yes ", " --> should focus on generate subterm | If No --> Should NOT focus since it is not worthy
        o_subterm, subterm = generate_subterm_query(q, list_synonym)
        qsq, qs_term, qs_term_adjust = generate_querystringquery_and_subquery(subterm, max_change)
        o_qsq, o_qs_term, o_qs_term_adjust = generate_querystringquery_and_subquery(o_subterm, max_change)
        queries_part += [create_json_query_string_part(query=o_qsq, field="description", boost=3)]
        queries_part += [create_json_query_string_part(query=qsq, field="description", boost=2)]
        queries_part += [create_json_query_string_part(query=o_qsq, field="description_clip", boost=1.25)]
        for sub_query, o_sub_query in zip(qs_term, o_qs_term):
            queries_part += [
                create_json_query_string_part(query=sub_query[0], field="description", boost=0.75),
                create_json_query_string_part(query=o_sub_query[0], field="description_clip", boost=0.45)
            ]
        for sub_query, o_sub_query in zip(qs_term_adjust, o_qs_term_adjust):
            queries_part += [
                create_json_query_string_part(query=sub_query[0], field="description", boost=0.35),
                create_json_query_string_part(query=o_sub_query[0], field="description_clip", boost=0.1)
            ]
    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "9.5*(cosineSimilarity(params.query_embedded, doc['description_embedded']) + 1.0)",
                "params": {"query_embedded": embedded_query}
            }
        }
    }

    queries_part += [script_query] + location_decay
    queries_part_string = str(queries_part)
    queries_part_string = queries_part_string.replace("'", "\"")
    queries_part_string = queries_part_string.replace("doc[\"description_embedded\"]", "doc['description_embedded']")
    queries_part_string = queries_part_string[1:-1]
    result += queries_part_string
    result += "],\"tie_breaker\":" + str(tie_breaker)
    result += "}},\"filter\":"
    filter_query = get_time_query(q)
    if having_location:
        filter_query += "&& doc['location'].size()!=0"
    result += json.dumps({
        "script": {
            "script": {
                "source": filter_query,
                "lang": "painless"
            }
        }
    })
    result += "}}}"
    request_string = result
    return request_string


def generate_query_text(q, max_change=1, tie_breaker=0.7, numb_of_result=100):
    '''
    Quite the same with generate_es_query_dismax but now inclide query string query, not multimatch anymore
    Generate elastic-formatted request and use the result for the input of elasticsearch
    list_synonym: list of synonym generated from Glove and only support classes in yolo or cbnet or your own defined
    max_change >= 0: See generate_near_query
    tie_breaker: See elasticsearch document
    Output:
        + request_string is the txt format of the elasticsearch formatted request
    '''
    q = q.lower()
    having_comma = q.find(",")
    if having_comma > 0:
        having_comma = True
    else:
        having_comma = False
    word_tokens = word_tokenize(q)
    good_tokens = [word for word in word_tokens if word not in stop_words]
    adjust_sentence_query = ' '.join(good_tokens)
    result = "{\"size\":" + str(numb_of_result)
    result += ",\"_source\": {\"includes\": [\"id\", \"description\", \"time\", \"location\", \"address\", \"nearby POI\", \"driving\", \"weekday\" ]}"

    result += ",\"query\":{\"dis_max\":{\"queries\":["
    queries_part = []
    queries_part += [create_json_query_string_part(query=adjust_sentence_query, field="description_clip", boost=5)]
    having_location, location_query = create_location_query(q, field="address", boost=24)
    if having_location:
        queries_part += [location_query]
    if having_comma:  # Yes ", " --> should focus on generate subterm | If No --> Should NOT focus since it is not worthy
        o_subterm, subterm = generate_subterm_query(q, list_synonym)
        qsq, qs_term, qs_term_adjust = generate_querystringquery_and_subquery(subterm, max_change)
        o_qsq, o_qs_term, o_qs_term_adjust = generate_querystringquery_and_subquery(o_subterm, max_change)
        queries_part += [create_json_query_string_part(query=o_qsq, field="description", boost=3)]
        queries_part += [create_json_query_string_part(query=qsq, field="description", boost=2)]
        queries_part += [create_json_query_string_part(query=o_qsq, field="description_clip", boost=1.25)]
        for sub_query, o_sub_query in zip(qs_term, o_qs_term):
            queries_part += [
                create_json_query_string_part(query=sub_query[0], field="description", boost=0.75),
                create_json_query_string_part(query=o_sub_query[0], field="description_clip", boost=0.45)
            ]
        for sub_query, o_sub_query in zip(qs_term_adjust, o_qs_term_adjust):
            queries_part += [
                create_json_query_string_part(query=sub_query[0], field="description", boost=0.35),
                create_json_query_string_part(query=o_sub_query[0], field="description_clip", boost=0.1)
            ]
    queries_part_string = str(queries_part)
    queries_part_string = queries_part_string.replace("'", "\"")
    queries_part_string = queries_part_string[1:-1]
    result += queries_part_string
    result += "],\"tie_breaker\":" + str(tie_breaker)
    result += "}}}"

    request_string = result
    return request_string


def find_descriptive_attribute_in_list_images(database, list_images):
    '''
    Find all descriptive attributes (or something relavant) of list_images in the database (description json file storing all information of all images)
    Descriptive value is defined as below output
    Output:
        + result_high is a list of most appearance (except the one that all images contain) attribute/environment
        + result_low is a list of least common (most distinctive value) and dont take attribute appearing once
    '''

    scene_list_images = [database[x]['scene_image'].split(', ') for x in list_images]

    list_scene = []
    for l in scene_list_images:
        list_scene.extend(l)

    scene_numb = Counter(list_scene)
    len_list_images = len(list_images)

    scene_appearance = list(scene_numb.keys())
    numb_appearance = np.array(list(scene_numb.values()))  # array of numb of appearance of each scence
    threshold_high = np.quantile(numb_appearance, 0.8)  # only keep 20% of most common
    threshold_low = np.quantile(numb_appearance, 0.2)  # only keep 20% least common --> likely that it will be one

    if threshold_high == len_list_images:  # if threshold = len(list_images) --> all images have that att --> useless! --> remove that value
        threshold_high = np.max(numb_appearance[np.where(numb_appearance < len_list_images)])

    if threshold_low == 1:  # if thershold = 1 --> only 1 picture contains that value --> seem to be useless and not disticntive --> find the 2nd smallest
        threshold_low = np.min(numb_appearance[np.where(numb_appearance > 1)])

    select_index_high = list(np.where((numb_appearance >= threshold_high) & (numb_appearance < len_list_images))[0])
    select_index_low = list(np.where((numb_appearance <= threshold_low) & (numb_appearance > 1))[0])

    result_high = []
    for x in select_index_high:
        result_high.append(scene_appearance[x])

    result_low = []
    for x in select_index_low:
        result_low.append(scene_appearance[x])

    return result_low, result_high


def add_folder_to_id_images(FolderPath, id_image):
    # Add link folder to the id_image (based on the first 10 characters)

    if type(id_image) is list:
        result = [FolderPath + x[:4] + "-" + x[4:6] + "-" + x[6:8] + "/" + x for x in id_image]
    else:
        result = id_image[:4] + "-" + id_image[4:6] + "-" + id_image[6:8] + "/" + id_image

    return result


def is_location(word):
    word = word.split()[-1]
    for synset in wordnet.synsets(word):
        ss = synset
        while True:
            if len(ss.hypernyms()) > 0:
                ss = ss.hypernyms()[0]
                if ss in [wordnet.synset('structure.n.01'),
                          wordnet.synset('facility.n.01'),
                          wordnet.synset('organization.n.01'),
                          wordnet.synset('location.n.01'),
                          wordnet.synset('way.n.06')]:
                    return True
            else:
                break
    return False


def get_places(input_query):
    places = []
    text = word_tokenize(input_query)
    tags = pos_tag(text)
    for i, (word, tag) in enumerate(tags):
        if is_location(word) or tag == "NNP":
            j = i - 1
            while j >= 0:
                if tags[j][1] not in ['NN', 'POS', 'NNP', 'JJ', 'DT', 'FW', 'JJR', 'JJS', 'NP', 'NPS', 'NNS']:
                    break
                j -= 1
            places.append(' '.join(text[j + 1: i + 1]))

        if word.lower() == "work":
            places.append("dublin city university")

    # filter
    new_places = []
    for place in sorted(places, key=lambda x: len(x), reverse=True):
        existed = False
        for p in new_places:
            if place in p:
                existed = True
                break
        if not existed:
            new_places.append(place)

    return new_places
