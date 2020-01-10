from elasticsearch import Elasticsearch

es = Elasticsearch([{"host": "localhost", "port": 9200}])
#
interest_index = "lsc2019_combined_text_bow"

es.indices.put_mapping(index=interest_index, body={
  "properties": {
    "id":  { "type": "keyword"}
  }
})

es.indices.close(interest_index)
es.indices.put_settings(index=interest_index, body={
    "analysis": {
        "filter": {
            "address_shingle": {
                "type": "shingle",
                "max_shingle_size": 3,
                "output_unigrams": False,
                "output_unigrams_if_no_shingles": False
            }
        },
        "analyzer": {
            "analyzer_gps_description": {
                "type": "custom",
                "tokenizer": "standard",  # remove 'and' and ','
                "filter": [
                    "address_shingle",
                    "lowercase",
                    "english_stop",
                    "english_possessive_stemmer",
                    "english_stemmer"
                ]
            }
        }
    }
})
es.indices.open(interest_index)

# import json
# import textdistance
# asigned_gps = json.load(open('data/u1_meta.json'))
# addresses = set([asigned_gps[image]["address"].lower() for image in asigned_gps])
# poi = json.load(open('data/poi.json'))
#
# print(textdistance.monge_elkan.distance("dublin city university".split(), "dublin city university (dcu)".split()))
# print(textdistance.monge_elkan.distance("dublin city university".split(), "the helix".split()))

# address_and_gps = {}
# for address in addresses:
#     if address == "":
#         continue
#     for image in asigned_gps:
#         if address == asigned_gps[image]["address"].lower() and not asigned_gps[image]["location"] is None:
#             address_and_gps[address] = (asigned_gps[image]["location"], 1)
#             break
#
# for p in poi:
#     address_and_gps[p["name"].lower()] = (p["location"], p["diameter"])
#
# # import geopy.distance
# # def distance(lt1, ln1, lt2, ln2):
# #     return (geopy.distance.distance([lt1, ln1], [lt2, ln2]).km)
# #
# # def sea():
# #     with open('data/sea.geojson') as f:
# #         coastline = json.load(f)
# #     for point in coastline['features'][0]['geometry']['coordinates'][0]:
# #         if distance(point[1], point[0], 53.33306, -6.24889) < 20:
# #             address_and_gps["the sea"] = ['location':[point[1], point[0]], 'name': "sea", "extra": "ocean coastline coast beach" }
#
# json.dump(address_and_gps, open("data/address_and_gps.json", "w"))