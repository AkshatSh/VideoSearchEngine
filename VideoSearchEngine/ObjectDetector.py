from config import OBJECT_DETECTION_TYPE
from ObjectDetection import (
    TinyYolo,
    Yolo,
)
from PIL import Image
import cv2
import numpy as np

# Main file for the bounding box object detection problem

# Main API is here, more files may be used for the implementation

def get_object_detector():
    '''
    return the version specified in the configuration to use
    e.g. if there is a basic one and a complex one, the configuration should be able
    to decide which one to use
    '''
    if OBJECT_DETECTION_TYPE == "TINY_YOLO":
        return TinyYolo.TinyYoloNet()
    elif OBJECT_DETECTION_TYPE == "YOLO":
        return Yolo.YoloNet()
    else:
        raise Exception("Unknown OBJECT_DETECTION_TYPE = {}".format(OBJECT_DETECTION_TYPE))

'''
Describe API supported here

get_object_detector returns an nn.Module

The main operation to use is forward on an image

Take a look at ObjectDetection/detector.py for common use cases
'''

if __name__ == "__main__":
    picture_name = "data/pics/football2.png"
    #picture_name = "data/resized2014/COCO_train2014_000000012996.jpg"
    #loaded_ims = [cv2.imread(picture_name)]
    loaded_ims = Image.open(picture_name)
    #loaded_ims = np.array(loaded_ims)
    loaded_ims = [np.array(loaded_ims)]
    loaded_ims = np.array(loaded_ims)
    model = Yolo.YoloNet()
    class_names, bboxes, lengths = model.get_bbox(loaded_ims)
    print(class_names)