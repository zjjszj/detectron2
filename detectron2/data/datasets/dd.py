import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
print(curPath)
rootPath = os.path.split(curPath)[0]
print(rootPath)


#print(os.path.isfile('//coco/annotations/instances_train2017.json'))