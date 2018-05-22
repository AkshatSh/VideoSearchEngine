'''
Using Obj2Text model as described here:

https://github.com/xuwangyin/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning

'''

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.autograd import Variable

class EncoderCNN(nn.Module):
    def __init__(self, embedded_size, bbox_model):
        super(EncoderCNN, self).__init__()
        self.bbox_model = bbox_model

        # todo update number to work with YOLO detections
        self.linear = nn.Linear(2428, embedded_size)

        # normalize the batch after the embedding
        self.batch_norm = nn.BatchNorm1d(embedded_size, momentum=0.01)

        # intialize the weights with the distribution defined
        self.init_weights()
    
    def init_weights(self):
        '''
        Initialize the weights using a normal distribution
        Intialize bias to be 0
        '''
        self.linear.weight.data.normal_(0.0, 0.2)
        self.linear.bias.data.fill_(0)
    
    def forward(self, x):
        with torch.no_grad():
            # the bounding box model has been pretrained
            # most likely
            # TODO(akshats): wrap this around a command line flag
            x = self.bbox_model.forward(x)

        # convert results of bounding box detection to a variable
        x = Variable(x.data)

        # reduce to a single row for each batch
        x = x.view(x.size(0), -1)

        # apply a linear layer
        x = self.linear(x)

        # normalize the results
        x = self.bn(x)
        return x

