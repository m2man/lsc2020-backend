import images.MyLibrary_v2 as mylib_v2

import json
import requests
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from datetime import datetime, timedelta


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
            j = i - 1
            while j >= 0:
                if tags[j][1] not in ['NN', 'POS', 'NNP', 'JJ', 'DT', 'FW', 'JJR', 'JJS', 'NP', 'NPS', 'NNS']:
                    break
                j -= 1
            places.append(' '.join(text[j + 1: i + 1]))

    place_query = " OR ".join(places)
    return place_query


def query_all(es):
    request_result, id_result = mylib.search_es(es, index="lsc2020", request={"query": {"match_all": {}}},
                                                percent_thres=0.5, max_len=100)
    return id_result


def group_time(results, min_value=5):
    grouped_results = []
    results = [(res, datetime.strptime(res[0]["time"], "%Y/%m/%d %H:%M:%S+00")) for res in results]
    sorted_time = sorted(results, key=lambda res: res[1])

    starting_index = 0
    for i in range(len(sorted_time) - 1):
        time_split = sorted_time[i + 1][1] - sorted_time[i][1]
        if time_split.seconds > 120 and i - starting_index > min_value:
            grouped_results.append([x[0] for x in sorted_time[starting_index: i + 1]])
            starting_index = i + 1

    if starting_index < len(sorted_time):
        grouped_results.append([x[0] for x in sorted_time[starting_index:]])

    final_grouped_with_scores = []
    for period in grouped_results:
        avg_scores = round(max([x[1] for x in period]), 2)
        final_grouped_with_scores.append((avg_scores, [x[0] for x in sorted(period, key=lambda x: x[1], reverse=True)]))

    final_grouped_with_scores = sorted(final_grouped_with_scores, key=lambda x: x[0], reverse=True)

    return final_grouped_with_scores


def get_autocomplete_query(text):
    split = text.rsplit(',', 1)
    if len(split) == 2:
        before_comma, after_comma = split
    else:
        before_comma, after_comma = "", text
    if before_comma != "":
        before_comma += ", "

    temp_text = word_tokenize(after_comma)
    if not temp_text:
        return before_comma, ""

    query = temp_text[-1]
    j = len(temp_text) - 1
    if j == 0:
        return query, before_comma
    tags = nltk.pos_tag(temp_text)
    while j > 0:
        j -= 1
        if tags[j][1] == "DT":
            return " ".join(temp_text[j:]), before_comma + " ".join(temp_text[:j])
        if tags[j][1] not in ['NN', 'POS', 'NNP', 'JJ', 'DT', 'FW', 'JJR', 'JJS', 'NP', 'NPS', 'NNS']:
            return " ".join(temp_text[j + 1:]), " ".join(temp_text[:j + 1])
    return " ".join(temp_text[j:]), before_comma + " ".join(temp_text[:j])


def es_bow(input_query):
    # query is the string query
    # List_synonym is the synonym file provided in the repo
    # output will be result including 2 information for each image (image_path and name of that image)
    data = mylib_v2.generate_query_combined(q=input_query)
    headers = {"Content-Type": "application/json"}
    response = requests.post("http://localhost:9200/lsc2019_combined_text_bow/_search", headers=headers, data=data)
    print(data)
    if response.status_code == 200:
        # stt = "Success"
        response_json = response.json()  # Convert to json as dict formatted
        id_images = [[d["_source"], d["_score"]] for d in response_json["hits"]["hits"]]
    else:
        id_images = []
    return group_time(id_images)


def ES_autocomplete(es, original_query):
    query, not_included_query = get_autocomplete_query(original_query)

    if len(query) <= 2:
        return [], original_query

    if len(query.split()) == 1:
        query_request = {
                            "_source": {
                                "includes": ["name"]
                            },
                            "query": {
                                "function_score": {
                                    "query": {
                                        "match_bool_prefix": {
                                            "name": query.lower()
                                        }
                                    },
                                    "functions": [
                                        {
                                            "script_score" : {
                                                "script": {
                                                    "source": "Math.log(1 + doc['weight'].value)"
                                                }
                                            },
                                            "weight": 10
                                        },
                                        {
                                            "exp": {
                                                "length": {
                                                    "origin": len(query),
                                                    "scale": 1,
                                                    "offset": 5,
                                                    "decay": 0.5
                                                }
                                            },
                                            "weight": 5
                                        }
                                    ],
                                    "score_mode": "avg"
                                }
                            }
                        }
        query_request_json = json.dumps(query_request)

        res = es.search(index="autosuggest", body=query_request_json,
                        size=4)

        id_result = [r["_source"]["name"] for r in res['hits']['hits']]

        return id_result, not_included_query

    else:
        query_request = {"suggest": {
                            "place-suggest": {
                                "prefix": query.lower(),
                                "completion": {
                                    "field": "suggest",
                                    "fuzzy": {
                                        "fuzziness": min(10, len(query) // 3),
                                        "transpositions": True,
                                        "prefix_length": 0
                                    },
                                    "size": 4,
                                }
                            }
                        }}

        query_request_json = json.dumps(query_request)

        res = es.search(index="autosuggest", body=query_request_json,
                        size=9999)  # Show all result (9999 results if possible)

        id_result = [r["_source"]["name"] for r in res['suggest']['place-suggest'][0]['options']]

        return id_result, not_included_query


def gps_search(es, bounds, images):

    query_request = {
        "_source": {
            "includes": ["id", "description", "time", "location", "address", "nearby POI", "driving"]
        },
        "query": {
            "bool": {
                "must": {
                    "match_all": {}
                },
                "filter": {
                    "geo_bounding_box": {
                        "location": {
                            "top_left": {"lon": float(bounds[0]), "lat": float(bounds[3])},
                            "bottom_right": {"lon": float(bounds[2]), "lat": float(bounds[1])}
                        }
                    }
                }
            }
        }
    }

    if images:
        ids = []
        for period in images:
            ids.extend([res["id"] for res in period[1]])
        image_filter = {
            "terms": {
                "id": ids
            }
        }
        query_request["query"]["bool"]["filter"] = [query_request["query"]["bool"]["filter"], image_filter]
    query_request_json = json.dumps(query_request)
    res = es.search(index="lsc2019_combined_text_bow", body=query_request_json, size=50)  # Show all result (9999 results if possible)
    id_result = [[r["_source"], r["_score"]] for r in res['hits']['hits']]

    return group_time(id_result)


def es_two_events(query, conditional_query, condition, time_limit=10):
    data = mylib_v2.generate_query_combined(q=query)
    headers = {"Content-Type": "application/json"}
    response = requests.post("http://localhost:9200/lsc2019_combined_text_bow/_search", headers=headers, data=data)
    response_json = response.json()  # Convert to json as dict formatted
    main_events = group_time([[d["_source"], d["_score"]] for d in response_json["hits"]["hits"]], 0)

    pairs = []
    for score, main_event in main_events:
        time1 = datetime.strptime(main_event[0]["time"], '%Y/%m/%d %H:%M:00+00')
        if condition == 'after':
            time_bound = time1 - timedelta(hours=min(10, time1.hour))
            new_query = f"{conditional_query} on May {time1.day} before {datetime.strftime(time1, '%H:%M')} after {datetime.strftime(time_bound, '%H:%M')}"
        else:
            time_bound = time1 + timedelta(hours=min(10, 23 - time1.hour))
            new_query = f"{conditional_query} on May {time1.day} after {datetime.strftime(time1, '%H:%M')} before {datetime.strftime(time_bound, '%H:%M')}"
        data = mylib_v2.generate_query_combined(q=new_query)
        headers = {"Content-Type": "application/json"}
        response = requests.post("http://localhost:9200/lsc2019_combined_text_bow/_search", headers=headers, data=data)
        response_json = response.json()  # Convert to json as dict formatted
        conditional_events = [d["_source"] for d in response_json["hits"]["hits"]][:10]
        if len(conditional_events) > 0:
            pairs.append((score, main_event + conditional_events))

    return pairs

if __name__ == "__main__":
    query = "2 cars"
    print(es_bow(query))
