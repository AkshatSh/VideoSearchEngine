import video_utils
import pickle
import socket
import sys
from _thread import start_new_thread
import threading
import time
import torch
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
import video_utils
import random


def thread_main(conn, captioner, count, host, port):
    # Accept the pickle file sent by VideoDistributer.py and write/cache to local copy.
    filename_recv = "/tmp/VideoSearchEngine/recievecluster:" + str(count) + "|" + "host:" + socket.gethostname() + "|" + "worker.pkl"
    if not os.path.exists(os.path.dirname(filename_recv)):
        os.makedirs(os.path.dirname(filename_recv))
    f_recv = open(filename_recv,'wb')
    print("Writing file {}".format(filename_recv))
    data = conn.recv(1024)
    while data:
        f_recv.write(data)
        data = conn.recv(1024)
    conn.close()
    f_recv.close()


    # De-pickle file to reconstruct array of images, manipulate as needed.
    f_recv = open(filename_recv,'rb')
    try:
        unpickled_data = pickle.load(f_recv)
    except Exception as e:
        print(e)
        unpickled_data = []
    f_recv.close()

    # Clean up pickle file, comment out to retain pickle files
    if os.path.isfile(filename_recv):
        try:
            os.remove(filename_recv)
        except OSError as e:  # if failed, report it back to the user
            print ("Error: %s - %s." % (e.filename, e.strerror))

    #TODO: Implement what needs to happen with the unpickled_data
    metadata = unpickled_data[0]
    unpickled_cluster_filename = metadata["file_name"]  # unpickled_data[0]
    unpickled_cluster_num = metadata["cluster_num"]  # unpickled_data[1]
    total_clusters = metadata["total_clusters"]
    unpickled_data = unpickled_data[1:]
    summaries = []
    frame_clusters = video_utils.group_semantic_frames(unpickled_data)
    for frame_cluster in tqdm.tqdm(frame_clusters):
        frames = random.choices(frame_cluster, k=10)
        for frame in frames:
            frame = np.array([np.array(frame)])
            if frame is torch.cuda.FloatTensor:
                frame = frame.cpu()
            caption = captioner.get_caption(frame)
            summaries.append(caption)
       
    # Pickle the array of summaries.
    summaries.insert(0, {"file_name": unpickled_cluster_filename, "cluster_num": unpickled_cluster_num, "total_clusters": total_clusters})
    # summaries.insert(0, unpickled_cluster_filename)
    # summaries.insert(1, unpickled_cluster_num)
    filename_send = "/tmp/VideoSearchEngine/sendcluster:" + str(count) + "|" + "host:" + str(host) + "|" + "worker.pkl"
    print("Writing file {}".format(filename_send))
    f_send = open(filename_send,'wb')
    pickle.dump(summaries, f_send)
    f_send.close()

    # Create a socket object
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Send pickle file over the network to server.
    print("Sending cluster " + str(unpickled_cluster_num) + " to collector: " + str(host) + ":" + str(port))
    s.connect((host, port))
    f_send = open(filename_send,'rb')
    data = f_send.read(1024)
    while (data):
        s.send(data)
        data = f_send.read(1024)
    f_send.close()
    s.close()

    # Clean up pickle file, comment out to retain pickle files
    if os.path.isfile(filename_send):
        try:
            os.remove(filename_send)
        except OSError as e:  # if failed, report it back to the user
            print ("Error: %s - %s." % (e.filename, e.strerror))
    
def load_necessary():
    captioner = ImageCaptioner()
    captioner.load_models()
    return captioner

'''
Usage:
python video_util_worker.py <localhost:port_to_listen_on> <host_to_send_to:port_to_send_to>
python VideoSearchEngine/video_util_worker.py localhost:24448 lobster.cs.washington.edu:1234
'''
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python video_util_worker.py <localhost:port_to_listen_on> <host_to_send_to:port_to_send_to>")
        exit()

    host = ''                 # Symbolic name meaning all available interfaces 
    port = int(sys.argv[1].split(':')[1])

    collector_host = sys.argv[2].split(':')[0]
    collector_port = int(sys.argv[2].split(':')[1])

    image_captioner = load_necessary()
    print("Worker started, listening on: " + socket.gethostname()  + ":" + str(port) + " sending to: " + collector_host + ":" + str(collector_port))
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    s.listen(5)
    count = 0
    while True:
        #establish connection with client
        conn, addr = s.accept()
        # Start new thread
        start_new_thread(thread_main, (conn, image_captioner, count, collector_host, collector_port,))
        count = count + 1
    s.close()
        

