from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 

def predict_transform(prediction, input_dim, anchors, num_classes, CUDA=False):
    batch_size = prediction.size(0)
    stride = input_dim // prediction.size(2)
    grid_size = input_dim // stride
    bounding_box_attributes = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bounding_box_attributes * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bounding_box_attributes)
    
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

    # Sigmoid the center_X, center_Y, and confidence
    for i in [0, 1, 4]:
        prediction[:, :, i] = torch.sigmoid(prediction[:, :, i])
    
    # Add the center offsets
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    
    x_y_offset = torch.cat([x_offset, y_offset], 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset

    # Apply the anhors to the dimensions of the bounding box
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()
    
    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)

    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    # apply the sigmoid function to the class scores
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))

    # resize predictions to size of image
    prediction[:,:,:4] *= stride

    return prediction



