from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np

from .parse_cfg import (
    parse_cfg,
    create_modules
)

from .util import (
    predict_transform
)


class DarkNet(nn.Module):
    def __init__(self, cfgfile):
        super(DarkNet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)
    
    def forward(self, x, CUDA):
        detections = []
        modules = self.blocks[1:]
        outputs = {} # caching outputs for skip connections and route layers

        # whether we have encountered a detection yet, if we have we will append
        # to the detections map, otherwie we have to intialize it
        write = 0
        for i, module in enumerate(modules):
            module_type = (module["type"])
            if module_type == "convolutional" or module_type == "upsample" or module_type == 'maxpool':
                x = self.module_list[i](x) # apply the layer to the current value
            
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - i
                
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
                
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
                    
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1, map2), 1)
            
            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i - 1] + outputs[i + from_]
            
            elif module_type == "yolo":
                anchors = self.module_list[i][0].anchors

                # input dimension
                input_dim = int(self.net_info["height"])
                num_classes = int(module["classes"])

                # Transform
                x = x.data 
                x = predict_transform(x, input_dim, anchors, num_classes, CUDA)
                if not write:
                    detections = x
                    write = 1

                else:
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x # save the current output for shortcut and route layers

        return detections 
    
    def load_weights(self, weightfile):
        # open weights file with proper permissions
        file_pointer = open(weightfile, "rb")
        header = np.fromfile(file_pointer, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(file_pointer, dtype=np.float32)

        # iterate over the weights file, and load the weights into the module
        ptr = 0  # keep track of where we are in the weights array
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            # only weights defined for conv layers
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                except:
                    batch_normalize = 0
                
                conv = model[0]

                if batch_normalize:
                    bn = model[1]
                    
                    # number of weights in the batch layer
                    num_bn_biases = bn.bias.numel()

                    # load the weights 
                    bn_biases = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # cast the loaded weights into dims of model weights
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # Copy the data to the model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.data.copy_(bn_running_mean)
                    bn.running_var.data.copy_(bn_running_var)
                else:
                    # Number of biases
                    num_biases = conv.bias.numel()


                    # load the weights
                    conv_biases = torch.from_numpy(weights[ptr : ptr + num_biases])
                    ptr += num_biases

                    # reshape the loaded weights according to the dims of hte model
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # copy the data to the layer
                    conv.bias.data.copy_(conv_biases)

                num_weights = conv.weight.numel()
                conv_weights = torch.from_numpy(weights[ptr : ptr + num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)


            


