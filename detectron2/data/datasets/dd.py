import sys
import os
import json


json_file = os.path.join('coco/annotations', "instances_testdev2017.json")
with open(json_file) as f:
    imgs_anns = json.load(f)
    print(imgs_anns)