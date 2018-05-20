from PIL import Image
from TinyYolo import TinyYoloNet

def main():
    m = TinyYoloNet()
    m.float()
    m.eval()
    m.load_weights("data/yolo-tiny.weights")
    print("loaded weights")

if __name__ == "__main__":
    main()