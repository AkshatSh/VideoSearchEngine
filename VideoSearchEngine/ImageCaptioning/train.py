# From PyTorch Image Captioning tutorial

import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN, LayoutEncoder
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

from image_caption_utils import (
    to_var
)

def main(args):
    torch.manual_seed(args.seed) # set the seed for random generation

    # make sure to update the seed for the GPU environment
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # set up model directory to save the state of the model
    if not os.path.exists(args.modeloath):
        os.makedirs(args.modelpath)
    
        # Image preprocessing
    # For normalization, see https://github.com/pytorch/vision#models
    transform = transforms.Compose([
        # transforms.RandomCrop(args.crop_size),
        # transforms.RandomHorizontalFlip(),
        transforms.Scale(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225))])
    # Load vocabulary wrapper.
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # build the data loader
    