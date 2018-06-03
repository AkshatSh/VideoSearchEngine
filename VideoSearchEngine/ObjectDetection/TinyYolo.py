'''
Citing this pytorch implementation of tiny yolo from: https://github.com/marvis/pytorch-yolo2/blob/master/models/tiny_yolo.py

Original YOLO: https://pjreddie.com/darknet/yolo/

'''

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .bbox_detector import get_bbox
import cv2
from .DarknetModels.darknet import DarkNet

class TinyYoloNet(DarkNet):
    def __init__(self):
        super(TinyYoloNet, self).__init__("cfg/yolov3-tiny.cfg")
        self.load_weights("data/yolov3-tiny.weights")
    
    def get_bbox(self, images):
        return get_bbox(self, images)

if __name__ == "__main__":
    # picture_name = "/Users/akshatshrivastava/Downloads/22140528_1610225452334984_1142797924_o.jpg" 
    picture_name = "data/pics/dog-cycle-car.png"
    loaded_ims = [cv2.imread(picture_name)] * 2
    model = TinyYoloNet()
    class_names, bboxes, lengths = model.get_bbox(loaded_ims)
    print(lengths)