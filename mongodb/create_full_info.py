import json
import datetime
import calendar

asigned_gps = json.load(open('data/u1_meta.json'))
object_description = json.load(open('data/u1_cbnet.json'))
scene_description = json.load(open('data/u1_place365.json'))

all_description = {}
for image, description in object_description.items():

    description1 = ", ".join([f"{count} {obj}" for (obj, count) in description.items()] + scene_description[image].split(", "))
    # description2 = []
    # for obj, count in description.items():
    #     d = [str(count)]
    #     for i in range(count - 1):
    #         d.append(obj)
    #     if count >= 3:
    #         d.append("many")
    #     d.append(obj)
    #     d = " ".join(d)
    #     description2.append(d)
    # description2 = ", ".join(description2)
    all_description[image] = {"id": image,
                              "scene": scene_description[image],
                              "description": description1,
                              "description_clip": description1}
    if image in asigned_gps:
        all_description[image].update({
                  "weekday": calendar.day_name[datetime.datetime.strptime(asigned_gps[image]["time"],
                                                                          "%Y/%m/%d %H:%M:00+00").weekday()],
                  "time": asigned_gps[image]["time"],
                  "location": asigned_gps[image]["location"],
                  "address": asigned_gps[image]["address"],
                  "activity": asigned_gps[image]["activity"]})
    else:
        print(image, "not in meta")
        all_description[image].update({
            "weekday": "Unknown",
            "time": None,
            "location": None,
            "address": "",
            "activity": ""})

json.dump(all_description, open("data/u1_full_info.json", "w"))