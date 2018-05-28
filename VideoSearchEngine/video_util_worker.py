import video_utils
import pickle
import socket
import sys
from _thread import start_new_thread
import threading
import time

file_lock = threading.Lock()
conn_lock = threading.Lock()
count_lock = threading.Lock()
count = 0

def thread_main(conn):
    file_lock.acquire()
    filename = "host" + host + "port" + str(port) + "workerPickleFile.pkl"
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

    #TODO: Implement what needs to happen with the unpickled_data
    count_lock.acquire()
    video_utils.export_video_frames(unpickled_data, "../frames/bunny_clip/port" + str(port) + "thread" + str(count) + "worker/")
    count_lock.release()

# host and port are set in workers.conf 
if __name__ == '__main__':
    host = sys.argv[1].split(':')[0]
    port = int(sys.argv[1].split(':')[1])
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))

    # Accept the pickle file sent by VideoDistributer.py and write/cache to local copy.
    s.listen(5)

    while True:
        #establish connection with client
        conn, addr = s.accept()
        print('Connected to :', addr[0], ':', addr[1])
        # Start new thread
        count_lock.acquire()
        count = count + 1
        count_lock.release()
        start_new_thread(thread_main, (conn,))

