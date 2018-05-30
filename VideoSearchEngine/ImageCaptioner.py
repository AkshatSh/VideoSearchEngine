import ObjectDetection.TinyYolo as TinyYolo
import ImageCaptioningYolo.train as image_train
import ImageCaptioningYolo.im_args as im_args
from ImageCaptioningYolo.build_vocab import Vocabulary


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
    model = TinyYolo.TinyYoloNet()
    image_train.main(im_args.get_arg_parse(), model)