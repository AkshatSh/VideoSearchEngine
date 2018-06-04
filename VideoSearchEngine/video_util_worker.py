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

file_lock = threading.Lock()
conn_lock = threading.Lock()

def thread_main(conn, captioner, count):
    # Accept the pickle file sent by VideoDistributer.py and write/cache to local copy.
    filename = "id:" + str(count) + "|" + "host:" + socket.gethostname() + "|" + "port:" + str(port) + "|" + "worker.pkl"
    f = open(filename,'wb')
    data = conn.recv(1024)
    while data:
        f.write(data)
        data = conn.recv(1024)
    f.close()

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

    #TODO: Implement what needs to happen with the unpickled_data
    count = unpickled_data[0]
    print(count)
    unpickled_data = unpickled_data[1:]
    #for frame in tqdm.tqdm(unpickled_data):
      # frame = np.array([np.array(frame)])
      # print(captioner.get_caption(frame))
    #sep = '\n'
    #conn.send(str.encode(str(count)+sep))
    #conn.close()


      

def load_necessary():
    captioner = ImageCaptioner()
    captioner.load_models()
    return captioner

# host and port are set in workers.conf 
if __name__ == '__main__':
    host = ''                 # Symbolic name meaning all available interfaces 
    port = int(sys.argv[1].split(':')[1])
    image_captioner = load_necessary()
    print("Worker started, listening on :" + str(host) + ":" + str(port))
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    s.listen(5)
    count = 0
    while True:
        #establish connection with client
        conn, addr = s.accept()
        # Start new thread
        start_new_thread(thread_main, (conn,image_captioner,count,))
        count = count + 1
    s.close()
        

