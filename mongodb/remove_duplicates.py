import json
import geopy.distance
def distance(lt1, ln1, lt2, ln2):
    return(geopy.distance.distance([lt1, ln1], [lt2, ln2]).km)

points_of_interest = json.load(open('data/poi.json'))

def find_point_of_interest(location):
    for point in points_of_interest:
        dist = distance(*location, *point["location"])
        if dist < 1:
            yield point, dist

# remove duplicates:
new_pois = []
poi_ids = set()
for point in points_of_interest:
    if point["id"] not in poi_ids:
        poi_ids.add(point["id"])
        new_pois.append(point)
    else:
        print("Duplicated")

print(len(points_of_interest))
print(len(new_pois))
json.dump(new_pois, open('reduced_poi.json', 'w'))

