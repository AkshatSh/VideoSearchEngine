from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np

from .layers import (
    MaxPoolStride1,
    EmptyLayer,
    DetectionLayer,
)

def parse_cfg(cfgfile):
    """
    Takes a configuration file
    
    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    
    """
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')                        # store the lines in a list
    lines = [x for x in lines if len(x) > 0]               # get read of the empty lines 
    lines = [x for x in lines if x[0] != '#']              # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]           # get rid of fringe whitespaces

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":               # This marks the start of a new block
            if len(block) != 0:          # If block is not empty, implies it is storing values of previous block.
                blocks.append(block)     # add it the blocks list
                block = {}               # re-init the block
            block["type"] = line[1:-1].rstrip()     
        else:
            key,value = line.split("=") 
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks

def create_modules(blocks):
    net_info = blocks[0]     #Captures the information about the input and pre-processing    
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, x in enumerate(blocks[1: ]):
        module = nn.Sequential()
        if (x["type"] == "convolutional"):
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                bias = True
                batch_normalize = 0

            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # Create and add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module("conv_{}".format(index), conv)

            # Add the batch norm layer if one exists
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{}".format(index), bn)
            
            # check the activation.
            # either Linear or LRELU 
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{}".format(index), activn)
            
        # if its an upsampling layer
        # use BiLinear2dUpsampling
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            module.add_module("upsample_{}".format(index), upsample)

        ## Route Layer
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')

            # start of a route
            start = int(x["layers"][0])

            try:
                end = int(x["layers"][1])
            
            except:
                end = 0
            
            # Positive annotation
            if start > 0:
                start = start - index 
            if end > 0:
                end = end - index 

            route = EmptyLayer()
            module.add_module("route_{}".format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]
        
        ## Short cut layer
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)
        
        # Yolo detection layer
        elif (x["type"] == "yolo"):
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)
        elif x['type'] == 'maxpool':
            pool_size = int(x['size'])
            stride = int(x['stride'])
            if stride > 1:
                model = nn.MaxPool2d(pool_size, stride)
            else:
                model = MaxPoolStride1()
            output_filters.append(prev_filters)
            module.add_module("MaxPool_{}".format(index), model)
            module_list.append(module)
            continue

        
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
    
    return (net_info, module_list)