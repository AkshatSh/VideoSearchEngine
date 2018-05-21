import urllib
import urllib.request
import os

def download_file_if_not_exists(url, path):
    if not os.path.exists(path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        urllib.request.urlretrieve(url, filename=path)
        print("downloaded {} from {}".format(path, url))
    else:
        print("Skipping download ... {} already exists".format(path))

def download_yolo_files():
    download_file_if_not_exists("http://pjreddie.com/media/files/yolo.weights", "data/yolo.weights")
    download_file_if_not_exists("https://pjreddie.com/media/files/yolov3-tiny.weights", "data/yolov3-tiny.weights")
    download_file_if_not_exists("https://pjreddie.com/media/files/yolov2-tiny.weights", "data/yolov2-tiny.weights")
    download_file_if_not_exists("https://pjreddie.com/media/files/yolo-tiny.weights", "data/yolo-tiny.weights")
    download_file_if_not_exists("https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg", "cfg/yolov3.cfg")

    download_file_if_not_exists("https://github.com/ayooshkathuria/pytorch-yolo-v3/raw/master/dog-cycle-car.png", "data/pics/dog-cycle-car.png")

    download_file_if_not_exists("http://pjreddie.com/media/files/yolov3.weights", "data/yolov3.weights")
    download_file_if_not_exists("https://raw.githubusercontent.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch/master/data/coco.names", "data/coco.names")
    download_file_if_not_exists("https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg", "cfg/yolov3-tiny.cfg")

    download_file_if_not_exists("https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch/raw/master/pallete", "data/pallete")

def main():
    download_yolo_files()

if __name__ == "__main__":
    main()