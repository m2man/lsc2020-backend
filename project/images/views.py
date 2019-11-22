from pymongo import MongoClient
from django.http import JsonResponse
import json
from django.views.decorators.csrf import csrf_exempt
import pickle
from elasticsearch import Elasticsearch
from images.query import ESearch

# Directory to images
Synonym_glove_all_file = "static/List_synonym_glove_all.pickle"
with open(Synonym_glove_all_file, "rb") as f:
    synonym = pickle.load(f)

es = Elasticsearch([{"host": "localhost", "port": 9200}])

@csrf_exempt
def images(request):
    # Load MongoDB
    # client = MongoClient()
    # # db = client.user1
    # client = MongoClient("mongodb+srv://alie:mrF6V4p32aOEayJX@user1-ielbg.mongodb.net/test?retryWrites=true&w=majority").user1
    # db = client.test
    # images = db.images

    # Get message
    message = json.loads(request.body.decode('utf-8'))
    # Calculations
    queryset = ESearch(es, synonym, message['query'])
    response = {'results': queryset,
                'error': None}

    # JSONize
    response = JsonResponse(response)
    response["Access-Control-Allow-Origin"] = "*"
    response["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    response["Access-Control-Allow-Credentials"] = "true"
    response["Access-Control-Allow-Headers"] = "X-Requested-With, Content-Type"
    return response

