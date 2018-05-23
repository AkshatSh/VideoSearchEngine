import video_utils

import pickle
import socket

# TODO: Find a better way to for the server to listen, for now just listen on this port
host = ''        # Symbolic name meaning all available interfaces
port = 12345     # Arbitrary non-privileged port
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((host, port))

# Accept the pickle file sent by VideoDistributer.py and write/cache to local copy.
s.listen(1)
conn, addr = s.accept()
print('Connected by', addr)
f = open('serverPickleFile.pkl','wb')
data = conn.recv(1024)
while data:
    f.write(data)
    data = conn.recv(1024)
conn.close()
f.close()

# De-pickle file to reconstruct array of images, manipulate as needed.
f = open('serverPickleFile.pkl','rb')
unpickled_data = pickle.load(f)
f.close()

#TODO: Implement what needs to happen with the unpickled_data
#video_utils.export_video_frames(unpickled_data, "../frames/bunny_clip/")

