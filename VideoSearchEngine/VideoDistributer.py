import video_utils
import argparse
import pickle
import socket
import time

# Main file for taking a video, and separating it into multiple chunks
# and distributing the work

# Main API is here, more files may be used for the implementation

def get_video_distributor():
    '''
    return the version specified in the configuration to use
    e.g. if there is a basic one and a complex one, the configuration should be able
    to decide which one to use
    '''
    return None

'''
Describe API supported here
'''

#TODO: This should be a coroutine or done asynchronously so it does not become a bottleneck.
def send_frames(frame_cluster):
    '''
    Given an array of frames send it to an listening server for further processing. Use pickle
    to serialize the array so it can be sent over the network.
    '''
    # Pickle the array of frames.
    f = open('clientPickleFile.pkl','wb')
    pickle.dump(frame_cluster, f)
    f.close()

    # TODO: Pass in host and port should as parameters, depends on how many machines are avaliable.
    s = socket.socket()         # Create a socket object
    host = socket.gethostname() # Get local machine name
    port = 12345                # Reserve a port for your service.

    # Send pickle file over the network to server.
    s.connect((host, port))
    f = open('clientPickleFile.pkl','rb')
    data = f.read(1024)
    while (data):
        s.send(data)
        data = f.read(1024)
    f.close()
    s.close()

#TODO: Look into whether or not a sequential approach is ok or not for this.
def distribute_frames(frames, clusterSize=10):
    '''
    Given an array of frames break into subarrays and send each subarray
    to some server for processing.
    '''
    cluster = []
    for frame in frames:
        cluster.append(frame);
        if len(cluster) % clusterSize == 0:
            print("Send data")
            #send_frames(cluster)
            cluster.clear()
    if len(cluster) != 0:
        print("Send data")
        #send_frames(cluster)

'''
Example Usage:
    python VideoDistributer.py --video_path ../clips/bunny_clip.mp4 --output_path ../frames/bunny_clip/
'''

#TODO: Add arguments to: only extract every nth frame, change width/height of captured frames, etc.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument("--video_path", help="path of the video",type=str, required=True)
    parser.add_argument("--output_path", help="output path of the video frames", type=str)
    args = parser.parse_args()        
    res = video_utils.get_frames_from_video(args.video_path)
    send_frames(res)
    #distribute_frames(res, len(res))

    # Uncomment if frames should be written to directory.
    #video_utils.export_video_frames(res, args.output_path)