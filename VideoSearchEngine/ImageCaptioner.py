import ObjectDetection.TinyYolo as TinyYolo
import ImageCaptioningYolo.train as image_train
import ImageCaptioningYolo.im_args as im_args
import ImageCaptioningYolo.sample as image_sample
from ImageCaptioningYolo.models import (
    EncoderCNN,
    YoloEncoder,
    DecoderRNN,
)
from ImageCaptioningYolo.build_vocab import Vocabulary
import ObjectDetection.Yolo as Yolo
import pickle
import torch
import numpy as np
from PIL import Image


def load_image(image_path):
    image = Image.open(image_path)
    image = image.resize([256, 256], Image.LANCZOS)
    
    image = np.array([np.array(image)])
    return image


class ImageCaptioner(object):

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pass
    
    def load_models(self):
        print("Beginning loading of Image Captioner Network")
        device = self.device
        self.tiny_yolo = TinyYolo.TinyYoloNet().to(device)
        self.yolo = Yolo.YoloNet().to(device)
        args = im_args.get_arg_parse()
        bbox_model = self.tiny_yolo
        with open(args.vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)

        self.encoder = EncoderCNN(args.embed_size).eval().to(device)
        
        self.yolo_encoder = YoloEncoder(
            args.layout_embed_size, 
            args.hidden_size, 
            bbox_model, 
            args.embed_size, 
            len(self.vocab), 
            self.vocab,
            args.num_layers
        ).to(device)

        self.decoder = DecoderRNN(
            args.embed_size,
            args.hidden_size,
            len(self.vocab),
            args.num_layers
        ).to(device)

        self.yolo_encoder.load_state_dict(torch.load(args.yolo_encoder_path, map_location=lambda storage, loc: storage))
        # yolo_encoder.bbox_model = other
        self.decoder.load_state_dict(torch.load(args.decoder_path, map_location=lambda storage, loc: storage))
        self.encoder.load_state_dict((torch.load(args.encoder_path, map_location=lambda storage, loc: storage)))

        print("Loaded Image Captioner Network")
    
    def get_caption(self, image):
        image_tensor = torch.Tensor(image).to(self.device)
        feature1 = self.encoder(image_tensor)
        feature = self.yolo_encoder(self.tiny_yolo, image_tensor).squeeze() 
        c = feature1 + feature
        sampled_ids = self.decoder.sample(c)
        sampled_ids = sampled_ids[0].cpu().numpy()

        sampled_caption = []
        for word_id in sampled_ids:
            word = self.vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(sampled_caption)

        return sentence



# Main file for generating text for an image

# Main API is here, more files may be used for the implementation

def get_image_captioner():
    '''
    return the version specified in the configuration to use
    e.g. if there is a basic one and a complex one, the configuration should be able
    to decide which one to use
    '''
    return None

'''
Describe API supported here
'''


if __name__ == "__main__":
    captioner = ImageCaptioner()
    captioner.load_models()
    # model = TinyYolo.TinyYoloNet()
    # other = Yolo.YoloNet() # TinyYolo.TinyYoloNet()
    # args = im_args.get_arg_parse()
    print(captioner.get_caption(load_image("data/pics/test_bunny_2.png")))
    # print(captioner.get_caption(load_image("data/pics/test_bunny_2.png")))
    # print(captioner.get_caption(load_image("data/pics/test_bunny_2.png")))
    # print(captioner.get_caption(load_image("data/pics/football2.png")))
    # image_train.main(im_args.get_arg_parse(), other)
    # image_sample.execute("data/pics/test_bunny_2.png", model, args, other)
    # image_sample.execute("data/pics/test_bunny.png", model, args, other)
    # image_sample.execute("data/pics/dog-cycle-car.png", model, args, other)
    # image_sample.execute("data/pics/jumping.png", model, args, other)
    # image_sample.execute("data/pics/talking.png", model, args, other)
    # image_sample.execute("data/pics/football.png", model, args, other)
    # image_sample.execute("data/pics/soccer.png", model, args, other)
    # image_sample.execute("data/pics/football2.png", model, args, other)