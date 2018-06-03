import ObjectDetection.TinyYolo as TinyYolo
import ImageCaptioningYolo.train as image_train
import ImageCaptioningYolo.im_args as im_args
import ImageCaptioningYolo.sample as image_sample
from ImageCaptioningYolo.build_vocab import Vocabulary
import ObjectDetection.Yolo as Yolo


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
    other = Yolo.YoloNet() # TinyYolo.TinyYoloNet()
    args = im_args.get_arg_parse()
    # image_train.main(im_args.get_arg_parse(), other)
    image_sample.execute("data/pics/test_bunny_2.png", model, args, other)
    image_sample.execute("data/pics/test_bunny.png", model, args, other)
    image_sample.execute("data/pics/dog-cycle-car.png", model, args, other)
    image_sample.execute("data/pics/jumping.png", model, args, other)
    image_sample.execute("data/pics/talking.png", model, args, other)
    image_sample.execute("data/pics/football.png", model, args, other)
    image_sample.execute("data/pics/soccer.png", model, args, other)
    image_sample.execute("data/pics/football2.png", model, args, other)