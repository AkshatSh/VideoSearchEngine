'''
Citing this pytorch implementation of tiny yolo from: https://github.com/marvis/pytorch-yolo2/blob/master/models/tiny_yolo.py

Original YOLO: https://pjreddie.com/darknet/yolo/

'''

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .bbox_detector import get_bbox

from .DarknetModels.darknet import DarkNet

class YoloNet(DarkNet):
    def __init__(self):
        super(YoloNet, self).__init__("cfg/yolov3.cfg")
        self.load_weights("data/yolov3.weights")

    def get_bbox(self, images):
        return get_bbox(self, images)