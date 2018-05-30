from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from .DarknetModels.util import *
import argparse
import os 
import os.path as osp
from .DarknetModels.darknet import DarkNet as Darknet
import pickle as pkl
import pandas as pd
import random

nms_thresh = 0.4
confidence = 0.5
batch_size = 1
CUDA = torch.cuda.is_available()
num_classes = 80    #For COCO

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

def get_model():
    return TinyYoloNet()

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """
    print("prepping images")
    img = cv2.resize(img, (inp_dim, inp_dim))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    print("prepped images")
    return img

classes = load_classes("data/coco.names")

def get_bbox(model, images):

    inp_dim = int(model.net_info["height"])
    #If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()

    #Set the model in evaluation mode
    model.eval()

    print("creating py torch variables")
    #PyTorch Variables for images
    im_batches = list(map(prep_image, images, [inp_dim for x in range(len(images))]))

    #List containing dimensions of original images
    im_dim_list = [(x.shape[1], x.shape[0]) for x in images]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)

    print("creating im dim list")

    if CUDA:
        im_dim_list = im_dim_list.cuda()

    leftover = 0
    if (len(im_dim_list) % batch_size):
        leftover = 1
    
    if batch_size != 1:
        num_batches = len(images) // batch_size + leftover            
        im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size,
                            len(im_batches))]))  for i in range(num_batches)]
    print("creating prediction shit")

    bboxes = []
    classes_detected = []
    class_names = []
    lengths = []
    for i, batch in enumerate(im_batches):
        if CUDA:
            batch = batch.cuda()
        
        prediction = model(Variable(batch, requires_grad = False), CUDA)
        prediction = write_results(prediction, confidence, num_classes, nms_conf = nms_thresh)
        # result has 
        # index in batch
        # 4 box coordinates
        # object scorness
        # the score for the class
        # maximum confidence 
        # and the index of the class
        curr_image_bbox = []
        curr_image_class = []
        if type(prediction) != int:
            curr_image_bbox = prediction[:,1:5] # [output[1:5] for output in prediction]
            curr_image_class = prediction[:, -1] # torch.FloatTensor([output[-1] for output in prediction])
            class_names.append([classes[int(id)] for id in curr_image_class])
        bboxes.append((bboxes, curr_image_bbox))
        classes_detected.append((classes_detected, curr_image_class))
        lengths.append(len(curr_image_class))

    return class_names, bboxes, lengths