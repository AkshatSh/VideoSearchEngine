import urllib
import urllib.request
import os
import zipfile

def download_file_if_not_exists(url, path):
    if not os.path.exists(path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        urllib.request.urlretrieve(url, filename=path)
        print("downloaded {} from {}".format(path, url))
    else:
        print("Skipping download ... {} already exists".format(path))

def unzip_all_files_in_director(dir_name):
    # dir_name = "data/"
    extension = ".zip"
    for item in os.listdir(dir_name): # loop through items in dir
        item = os.path.join(dir_name, item)
        if item.endswith(extension): # check for ".zip" extension
            print("unzipping {} ...".format(item))
            file_name = os.path.abspath(item) # get full path of files
            zip_ref = zipfile.ZipFile(file_name) # create zipfile object
            zip_ref.extractall(dir_name) # extract file to dir
            zip_ref.close() # close file
            os.remove(file_name) # delete zipped file
            print("finished unzipping {} ...".format(item))
    
def download_tacos_dataset(shouldDownloadLarge=False):
    download_file_if_not_exists("http://datasets.d2.mpi-inf.mpg.de/MPII-Cooking-2/TACoS-Multi-Level-1.0.zip", "data/tacos.zip")

    if shouldDownloadLarge:
        download_file_if_not_exists(
            "http://datasets.d2.mpi-inf.mpg.de/MPII-Cooking-2/MPII-Cooking-2-videos.tar.gz", 
            "data/cooking-videos.tar.gz"
        )

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

    download_file_if_not_exists("https://raw.githubusercontent.com/marvis/pytorch-yolo2/master/cfg/tiny-yolo-voc.cfg", "cfg/tiny-yolo-voc.cfg")

    download_file_if_not_exists("https://github.com/leetenki/YOLOtiny_v2_chainer/raw/master/tiny-yolo-voc.weights", "data/tiny-yolo-voc.weights")

    download_file_if_not_exists("https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov1-tiny.cfg", "cfg/yolo-tiny.cfg")
    download_file_if_not_exists("http://pjreddie.com/media/files/yolo-tiny.weights", "data/yolo-tiny.weights")

def main():
    download_yolo_files()
    download_tacos_dataset()
    unzip_all_files_in_director("data/")

if __name__ == "__main__":
    main()