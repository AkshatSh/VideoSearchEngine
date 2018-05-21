'''
Citing this pytorch implementation of tiny yolo from: https://github.com/marvis/pytorch-yolo2/blob/master/models/tiny_yolo.py

Original YOLO: https://pjreddie.com/darknet/yolo/

'''

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from DarknetModels.darknet import DarkNet

class TinyYoloNet(DarkNet):
    def __init__(self):
        super(TinyYoloNet, self).__init__("cfg/yolov3-tiny.cfg")
        self.load_weights("data/yolov3-tiny.weights")