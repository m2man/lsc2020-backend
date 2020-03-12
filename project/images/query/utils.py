import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np
import geopy.distance
import requests

COMMON_PATH = os.getenv("COMMON_PATH")
group_info = json.load(open(f"{COMMON_PATH}/group_info.json"))


def distance(lt1, ln1, lt2, ln2):
    return (geopy.distance.distance([lt1, ln1], [lt2, ln2]).km)


def post_request(json_query, index="lsc2019_combined_text_bow"):
    headers = {"Content-Type": "application/json"}
    response = requests.post(
        f"http://localhost:9200/{index}/_search", headers=headers, data=json_query)
    if response.status_code == 200:
        # stt = "Success"
        response_json = response.json()  # Convert to json as dict formatted
        id_images = [[d["_source"], d["_score"]]
                     for d in response_json["hits"]["hits"]]
    else:
        print('Wrong')
        # print(json_query)
        print(response.status_code)
        id_images = []
    return id_images


def find_place_in_available_group(regrouped_results, new_group, time_limit=0.5):
    begin_time, end_time = get_time_of_group([res[0] for res in new_group])
    if regrouped_results:
        for regroup in regrouped_results:
            if abs(begin_time - regrouped_results[regroup]["begin_time"]) < timedelta(hours=time_limit) and \
                    abs(end_time - regrouped_results[regroup]["end_time"]) < timedelta(hours=time_limit):
                begin_time = min(
                    begin_time, regrouped_results[regroup]["begin_time"])
                end_time = max(
                    end_time, regrouped_results[regroup]["end_time"])
                return regroup, begin_time, end_time
    return "", begin_time, end_time


def get_before_after(images):
    min_group = np.argmin([int(image["group"].split('_')[-1])
                           for image in images])
    max_group = np.argmax([int(image["group"].split('_')[-1])
                           for image in images])
    return images[min_group]["before"], images[max_group]["after"]


def group_results(results, get_time_bound=False, group_time=0, factor="group"):
    grouped_results = defaultdict(lambda: [])
    for result in results:
        group = result[0][factor]
        grouped_results[group].append(result)

    # Group again for hours < 2h, same location
    regrouped_results = {}
    count = 0
    for group in grouped_results:
        new_group = grouped_results[group]
        regroup, begin_time, end_time = find_place_in_available_group(
            regrouped_results, new_group, group_time)
        if regroup and factor == "group":
            regrouped_results[regroup]["raw_results"].extend(new_group)
            regrouped_results[regroup]["begin_time"] = begin_time
            regrouped_results[regroup]["end_time"] = end_time
        else:
            count += 1
            regrouped_results[f"group_{count}"] = {"raw_results": grouped_results[group],
                                                   "begin_time": begin_time,
                                                   "end_time": end_time}

    sorted_groups = []
    for group in regrouped_results:
        sorted_with_scores = sorted(
            regrouped_results[group]["raw_results"], key=lambda x: x[1], reverse=True)
        score = sorted_with_scores[0][1]
        sorted_groups.append((score,
                              [res[0] for res in sorted_with_scores],
                              regrouped_results[group]["begin_time"],
                              regrouped_results[group]["end_time"]))

    sorted_groups = sorted(sorted_groups, key=lambda x: x[0], reverse=True)

    final_results = []

    for score, images, begin_time, end_time in sorted_groups:
        final_results.append({
            "current": [image["image_path"] for image in images],
            "before": images[0]["before"],
            "after": images[0]["after"],
            "begin_time": begin_time if get_time_bound else '',
            "end_time": end_time if get_time_bound else ''})
    return final_results


def get_time_of_group(images):
    times = [datetime.strptime(
        image["time"], "%Y/%m/%d %H:%M:%S+00") for image in images]
    begin_time = min(times)
    end_time = max(times)
    return begin_time, end_time
