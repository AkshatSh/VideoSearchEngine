# From PyTorch Image Captioning tutorial

import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from image_data_loader import get_loader
from build_vocab import Vocabulary
from LanguageModels import EncoderCNN, DecoderRNN, LayoutEncoder
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from tqdm import (
    tqdm,
    trange
)

from image_caption_utils import (
    to_var
)

def main(args):
    torch.manual_seed(args.seed) # set the seed for random generation

    # make sure to update the seed for the GPU environment
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # set up model directory to save the state of the model
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
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
    data_loader = get_loader(args.image_dir, args.caption_path, vocab, args.coco_detection_result,
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)
    
    # Build the CNN encoder (using CNN features)
    encoder = EncoderCNN(args.embed_size)

    # Build the Layout Encoder (using features from bbox results)
    # the layout encoder hidden state size must be the same with decoder input size
    layout_encoder = LayoutEncoder(args.layout_embed_size, args.embed_size, 100, args.num_layers)
    decoder = DecoderRNN(args.embed_size, args.hidden_size,
                         len(vocab), args.num_layers)
    
    if torch.cuda.is_available():
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        layout_encoder = layout_encoder.cuda()
    
    criterion = nn.CrossEntropyLoss()
    params =  (
        list(layout_encoder.parameters()) + 
        list(decoder.parameters()) + 
        list(encoder.linear_parameters()) + 
        list(encoder.bn.parameters.list)
    )

    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Train the models

    # calculate total size
    total_step = len(data_loader)
    for epoch in trange(args.num_epochs):
        for i, (images, captions, lengths, label_seqs, location_seqs, layout_lengths) in enumerate(tqdm(data_loader)):
            # lets create the necessary variables
            images = to_var(images, volatile=True)
            captions = to_var(captions)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            # Forward pass

            # zero out the gradients
            layout_encoder.zero_grad()
            decoder.zero_grad()
            encoder.zero_grad()

            # forward pass of the network
            features = encoder(images)
            layout_features = layout_encoder(label_seqs, location_seqs, layout_lengths)
            combined_features = features + layout_features

            outputs = decoder(combined_features, captions, lengths)

            # compute the loss
            loss = criterion(outputs, targets)
            
            # backprop
            loss.backward()
            optimizer.step()


            # log info
            if i % args.log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                      % (epoch, args.num_epochs, i, total_step,
                         loss.data[0], np.exp(loss.data[0])))

            # Save the models
            if (i + 1) % args.save_step == 0:
                torch.save(decoder.state_dict(),
                           os.path.join(args.model_path,
                                        'decoder-%d-%d.pkl' % (epoch + 1, i + 1)))
                torch.save(encoder.state_dict(),
                           os.path.join(args.model_path,
                                        'encoder-%d-%d.pkl' % (epoch + 1, i + 1)))

# Build Arg Parser
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/',
                        help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='../../data/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='../../data/resized2014',
                        help='directory for resized images')
    parser.add_argument('--caption_path', type=str,
                        default='../../data/annotations/captions_train2014.json',
                        help='path for train annotation json file')
    parser.add_argument('--coco_detection_result', type=str,
                        default='../../data/coco_yolo_objname_location.json',
                        help='path coco object detection result file')
    parser.add_argument('--log_step', type=int, default=10,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1000,
                        help='step size for saving trained models')

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=256,
                        help='dimension of word embedding vectors')
    parser.add_argument('--layout_embed_size', type=int, default=256,
                        help='layout encoding size')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=123, help='random generator seed')
    args = parser.parse_args()
    print(args)
    main(args)
