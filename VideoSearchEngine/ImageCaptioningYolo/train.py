import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os
import pickle
from .data_loader import get_loader 
from .build_vocab import Vocabulary
from .models import EncoderCNN, DecoderRNN, DecoderLayoutRNN, YoloEncoder
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from tqdm import (
    tqdm,
    trange
)

from .im_args import (
    get_arg_parse
)

os.environ["CUDA_VISIBLE_DEVICES"]="0"


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(encoder, decoder, data_loader, step_count, tensor_board_writer):
    criterion = nn.CrossEntropyLoss()
    loss_total = 0
    loss_count = 0
    for i, (images, captions, lengths) in enumerate(tqdm(data_loader)):
        # Set mini-batch dataset
        images = Variable(images, requires_grad=False).to(device)
        # print(images.shape)
        captions = captions.to(device)
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

        features = encoder(images)
        outputs = decoder(features, captions, lengths)
        loss = criterion(outputs, targets)
        decoder.zero_grad()
        encoder.zero_grad()
        loss_total += loss.item()
        loss_count += 1
        if torch.cuda.is_available():
                torch.cuda.empty_cache()
    tensor_board_writer.scalar_summary("dev_loss", float(loss_total) / loss_count, step_count)

def main(args, bbox_model):
    tensor_board_writer = None # Logger()
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([ 
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    with open(args.test_vocab_path, 'rb') as f:
        test_vocab = pickle.load(f)
    
    # Build data loader
    data_loader = get_loader(args.image_dir, args.caption_path, vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers) 
    
    test_data_loader = get_loader(
        args.test_image_dir, 
        args.test_caption_path, 
        test_vocab, 
        transform, 
        args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers)

    # Build the models
    encoder = EncoderCNN(args.embed_size).to(device)

    yolo_encoder = YoloEncoder(
        args.layout_embed_size, 
        args.hidden_size, 
        bbox_model, 
        args.embed_size, 
        len(vocab), 
        vocab
    )

    decoder = DecoderLayoutRNN(
        args.embed_size,
        args.hidden_size,
        len(vocab),
        args.num_layers
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = (
        list(decoder.parameters()) + 
        list(encoder.linear.parameters()) + 
        list(encoder.bn.parameters()) +
        list(yolo_encoder.parameters())
    )

    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    total_step = len(data_loader)
    
    for epoch in trange(args.num_epochs):
        test(encoder, decoder, test_data_loader, (epoch) * total_step, tensor_board_writer)
        for i, (images, captions, lengths) in enumerate(tqdm(data_loader)):
            images = images.to(device)
            # print(images.shape)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            # Forward, backward and optimize
            features = encoder(images)
            yolo_features = yolo_encoder(images)
            combined_features = yolo_features + features
            outputs = decoder(combined_features, captions, lengths)
            loss = criterion(outputs, targets)

            break