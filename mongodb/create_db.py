import pymongo
import base64
import pprint
import os
from datetime import datetime
from pymongo import MongoClient
from bson import json_util
import json


client = MongoClient("mongodb+srv://alie:mrF6V4p32aOEayJX@user1-ielbg.mongodb.net/test?retryWrites=true&w=majority")
db = client.test
images = db.images

db.images.create_index([('name', pymongo.ASCENDING)], unique=True)

for image in sorted(os.listdir('images')):
    if image in ['meta', '.DS_Store']:
        continue
    time = datetime.strptime(image, '%Y%m%d_%H%M%S_000.jpg')
    image_info = {'name': image,
                  'time': time.strftime('%Y/%m/%d %H:%M:%S+00'),
                  }
    images.insert_one(image_info)


print(json.dumps(list(images.find()), default=json_util.default))
# image = images.find_one({'name': '20160808_055120_000.jpg'})
# imagedata = base64.b64decode(image['image'])
# with open('image.jpg', 'wb') as f:
#     f.write(imagedata)
