import json
import geopy.distance
from collections import defaultdict
from tqdm import tqdm

old_gps = json.load(open('more_gps.json'))


def distance(lt1, ln1, lt2, ln2):
    return (geopy.distance.distance([lt1, ln1], [lt2, ln2]).km)


points_of_interest = json.load(open('poi.json'))

poi_dicts = {}
for point in points_of_interest:
    poi_dicts[point["id"]] = point["name"]


def find_point_of_interest(location):
    for point in points_of_interest:
        dist = distance(*location, *point["location"])
        if dist < 1:
            yield point, dist


gps_points = defaultdict(lambda: 0)
asigned_gps = json.load(open('asigned_gps.json'))
for image in tqdm(asigned_gps):
    for point, dist in find_point_of_interest(asigned_gps[image]["location"]):
        gps_points[point["name"]] += 1 - dist

json.dump(gps_points, open('autosuggest_poi.json', 'w'))

for *_, lt, ln in tqdm(old_gps):
    if [lt, ln] != [None, None]:
        for point, dist in find_point_of_interest([lt, ln]):
            gps_points[point["name"]] += 1 - dist

json.dump(gps_points, open('autosuggest_poi.json', 'w'))
