import video_utils
import pickle
import socket
import sys
from _thread import start_new_thread
import threading
import time
import os
import numpy as np
import tqdm
import ObjectDetection.TinyYolo as TinyYolo
import ImageCaptioningYolo.train as image_train
import ImageCaptioningYolo.im_args as im_args
import ImageCaptioningYolo.sample as image_sample
from ImageCaptioningYolo.build_vocab import Vocabulary
import ObjectDetection.Yolo as Yolo
from ImageCaptioner import ImageCaptioner

def thread_main(conn, count):
    # Accept the pickle file sent by VideoDistributer.py and write/cache to local copy.
    filename = "count:" + str(count) + "|" + "host:" + socket.gethostname() + "|" + "port:" + str(port) + "|" + "collector.pkl"
    f = open(filename,'wb')
    data = conn.recv(1024)
    while data:
        f.write(data)
        data = conn.recv(1024)
    f.close()
    conn.close()

    # De-pickle file to reconstruct array of images, manipulate as needed.
    f = open(filename,'rb')
    try:
        unpickled_data = pickle.load(f)
    except Exception as e:
        print(e)
        unpickled_data = []
    f.close()

    # Clean up pickle file, comment out to retain pickle files
    if os.path.isfile(filename):
        try:
            os.remove(filename)
        except OSError as e:  # if failed, report it back to the user
            print ("Error: %s - %s." % (e.filename, e.strerror))
    
    #cluster_num = unpickled_data[0]
    #unpickled_data = unpickled_data[1:]
    print(len(unpickled_data))

'''
Usage:
python VideoColleector.py <localhost:port_to_listen_on>
python VideoSearchEngine/VideoCollector.py localhost:24448 
'''
if __name__ == '__main__':
    if len(sys.argv) != 2:
      print("Usage: python video_util_worker.py <localhost:port_to_listen_on> ")
      exit()

    host = '' # Symbolic name meaning all available interfaces 
    port = int(sys.argv[1].split(':')[1])
    print("Collector started, listening on: " + socket.gethostname() + ":" + str(port))
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    s.listen(64)
    count = 0
    while True:
        #establish connection with client
        conn, addr = s.accept()
        # Start new thread
        start_new_thread(thread_main, (conn, count,))
        count = count + 1;
    s.close()
        