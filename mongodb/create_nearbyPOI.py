import geopy.distance
import json
from tqdm import tqdm


def distance(lt1, ln1, lt2, ln2):
    return (geopy.distance.distance([lt1, ln1], [lt2, ln2]).km)


points_of_interest = json.load(open('data/poi.json'))
descriptions = {}
asigned_gps = json.load(open('data/u1_meta.json'))


def find_point_of_interest(location):
    for point in points_of_interest:
        dist = distance(*location, *point["location"])
        if dist < 1:
            yield point, dist


for i, (image, gps_info) in tqdm(enumerate(asigned_gps.items()), total=len(asigned_gps)):
    description = []
    if "Dublin" in gps_info["region"]:
        if gps_info["location"]:
            nearby_POIs = sorted(list(find_point_of_interest(gps_info["location"])), key=lambda x: x[1])[:3]

            for poi in nearby_POIs:
                description.append(poi)
                # for i in range(len(poi[0]["name"].split())):
                #     description.append(" ".join(poi[0]["name"].split()[-i - 1:]))

    descriptions[image] = description
    if i % 500 == 0:
        json.dump(descriptions, open("data/u1_nearbyPOI.json", 'w'))

json.dump(descriptions, open("data/u1_nearbyPOI.json", 'w'))
