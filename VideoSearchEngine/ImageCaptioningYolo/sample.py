import os
from torchvision import transforms 
import torch
import pickle
from .build_vocab import Vocabulary
from .models import (
    YoloEncoder,
    DecoderLayoutRNN
)
from .im_args import get_arg_parse
import numpy as np
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    
    image = np.array(image)
    return image

def get_caption(image, bbox_model, args):
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    yolo_encoder = YoloEncoder(
        args.layout_embed_size, 
        args.hidden_size, 
        bbox_model, 
        args.embed_size, 
        len(vocab), 
        vocab
    ).to(device)

    decoder = DecoderLayoutRNN(
        args.embed_size,
        args.hidden_size,
        len(vocab),
        args.num_layers
    ).to(device)

    yolo_encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))
    image_tensor = torch.Tensor(image).to(device)

    feature = yolo_encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()

    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)

    print(sentence)


def execute(image_path, bbox_model, args):
    image = load_image(image_path)
    get_caption(image, bbox_model, args)
