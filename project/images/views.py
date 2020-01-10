from pymongo import MongoClient
from django.http import JsonResponse
import json
from django.views.decorators.csrf import csrf_exempt
import pickle
from elasticsearch import Elasticsearch
from images.query import ES_autocomplete, gps_search, es_bow, es_two_events

# Directory to images
Synonym_glove_all_file = "static/List_synonym_glove_all.pickle"
with open(Synonym_glove_all_file, "rb") as f:
    synonym = pickle.load(f)

es = Elasticsearch([{"host": "localhost", "port": 9200}])

@csrf_exempt
def images(request):
    # Get message
    message = json.loads(request.body.decode('utf-8'))
    # Calculations
    if ';' not in message['query']:
        queryset = es_bow(message['query'])
        response = {'results': queryset,
                    'error': None}
    else:
        main_query, conditional_query, condition =  message['query'].split(';')
        message = json.loads(request.body.decode('utf-8'))
        # Calculations
        queryset = es_two_events(main_query, conditional_query, condition)

        response = {'results': queryset,
                    'error': None}

    # JSONize
    response = JsonResponse(response)
    response["Access-Control-Allow-Origin"] = "*"
    response["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    response["Access-Control-Allow-Credentials"] = "true"
    response["Access-Control-Allow-Headers"] = "X-Requested-With, Content-Type"
    return response


@csrf_exempt
def autocomplete(request):
    # Get message
    message = json.loads(request.body.decode('utf-8'))
    # Calculations
    queryset, not_included_query = ES_autocomplete(es, message['query'])

    response = {'results': queryset,
                'not_included_query': not_included_query,
                'error': None}

    # JSONize
    response = JsonResponse(response)
    response["Access-Control-Allow-Origin"] = "*"
    response["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    response["Access-Control-Allow-Credentials"] = "true"
    response["Access-Control-Allow-Headers"] = "X-Requested-With, Content-Type"
    return response

@csrf_exempt
def gpssearch(request):
    # Get message
    message = json.loads(request.body.decode('utf-8'))
    # Calculations
    images = message["images"] if "images" in message else []
    queryset = gps_search(es, message['query'], images)
    response = {'results': queryset,
                'error': None}
    # JSONize
    response = JsonResponse(response)
    response["Access-Control-Allow-Origin"] = "*"
    response["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    response["Access-Control-Allow-Credentials"] = "true"
    response["Access-Control-Allow-Headers"] = "X-Requested-With, Content-Type"
    return response

@csrf_exempt
def dual_events(request):
    # Get message
    message = json.loads(request.body.decode('utf-8'))
    # Calculations
    queryset = es_two_events(message['main_query'], message['conditional_query'], message['condition'])

    response = {'results': queryset,
                'error': None}

    # JSONize
    response = JsonResponse(response)
    response["Access-Control-Allow-Origin"] = "*"
    response["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    response["Access-Control-Allow-Credentials"] = "true"
    response["Access-Control-Allow-Headers"] = "X-Requested-With, Content-Type"
    return response