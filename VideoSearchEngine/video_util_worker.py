import video_utils
import pickle
import socket
import sys
from _thread import start_new_thread
import threading
import time
import os
import ObjectDetection.TinyYolo as TinyYolo
import ImageCaptioningYolo.train as image_train
import ImageCaptioningYolo.im_args as im_args
import ImageCaptioningYolo.sample as image_sample
from ImageCaptioningYolo.build_vocab import Vocabulary
import ObjectDetection.Yolo as Yolo
from ImageCaptioner import ImageCaptioner

file_lock = threading.Lock()
conn_lock = threading.Lock()
count_lock = threading.Lock()
count = 0

def thread_main(conn):
    # Accept the pickle file sent by VideoDistributer.py and write/cache to local copy.
    file_lock.acquire()
    filename = "id:" + str(count) + "|" + "host:" + host + "|" + "port:" + str(port) + "|" + "worker.pkl"
    f = open(filename,'wb')
    file_lock.release()
    conn_lock.acquire()
    data = conn.recv(1024)
    while data:
        f.write(data)
        data = conn.recv(1024)
    conn.close()
    conn_lock.release()
    file_lock.acquire()
    f.close()
    file_lock.release()

    # De-pickle file to reconstruct array of images, manipulate as needed.
    file_lock.acquire()
    f = open(filename,'rb')
    try:
        unpickled_data = pickle.load(f)
    except Exception as e:
        print(e)
        unpickled_data = []
    f.close()
    file_lock.release()

    # Clean up pickle file, comment out to retain pickle files
    if os.path.isfile(filename):
        try:
            os.remove(filename)
        except OSError as e:  # if failed, report it back to the user
            print ("Error: %s - %s." % (e.filename, e.strerror))

    #TODO: Implement what needs to happen with the unpickled_data
    count_lock.acquire()
    print(len(unpickled_data))
    #video_utils.export_video_frames(unpickled_data, "../frames/bunny_clip/port" + str(port) + "thread" + str(count) + "worker/")
    count_lock.release()

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
    while True:
        #establish connection with client
        conn, addr = s.accept()
        # Start new thread
        count_lock.acquire()
        count = count + 1
        count_lock.release()
        start_new_thread(thread_main, (conn,))
    s.close()
        

