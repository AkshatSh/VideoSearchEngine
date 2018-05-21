from PIL import Image
from TinyYolo import TinyYoloNet
from darknet import DarkNet
from parse_cfg import (
    parse_cfg,
    create_modules
)
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np

import cv2

def get_test_input():
    img = cv2.imread("data/pics/dog-cycle-car.png")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ = img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_

def main():
    m = TinyYoloNet()
    m.float()
    m.eval()
    m.load_weights("data/yolo-tiny.weights")
    print("loaded weights")

if __name__ == "__main__":
    # blocks = parse_cfg("cfg/yolov3.cfg")
    # print(create_modules(blocks))
    # main()

    model = DarkNet("cfg/yolov3.cfg")
    model.load_weights("data/yolov3.weights")
    inp = get_test_input()
    pred = model(inp, torch.cuda.is_available())
    print(pred)