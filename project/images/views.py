import json

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from images.query import es, get_timeline


def jsonize(response):
    # JSONize
    response = JsonResponse(response)
    response["Access-Control-Allow-Origin"] = "*"
    response["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    response["Access-Control-Allow-Credentials"] = "true"
    response["Access-Control-Allow-Headers"] = "X-Requested-With, Content-Type"
    return response


@csrf_exempt
def images(request):
    # Get message
    message = json.loads(request.body.decode('utf-8'))
    # Calculations
    queryset = es(message['query'], message["gps_bounds"])[:100]
    response = {'results': queryset}
    return jsonize(response)


@csrf_exempt
def timeline(request):
    # Get message
    message = json.loads(request.body.decode('utf-8'))
    timeline = get_timeline(message['images'], message["timeline_type"])
    response = {'timeline': timeline}
    return jsonize(response)


@csrf_exempt
def gpssearch(request):
    # Get message
    message = json.loads(request.body.decode('utf-8'))
    # Calculations
    images = message["scenes"]
    display_type = message["display_type"]
    queryset = es_gps(es, message['query'], images, display_type)
    response = {'results': queryset,
                'error': None}
    return jsonize(response)
