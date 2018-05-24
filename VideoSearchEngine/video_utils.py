import cv2
import numpy as np
import os
from tqdm import tqdm
'''
This util file is for anything related to video processing that can be factored out into here
'''

def get_frames_from_video(video_path):
    '''
    Given a video path, read the video, store all the frames in the array
    and return it.
    '''
    # Playing video from file:
    frameArray = []
    cap = cv2.VideoCapture(video_path)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Break if no image is returned (have reached end of video)
        if frame is None or ret is False:
            break

        # Add frame to array
        frameArray.append(frame)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    return frameArray

def export_video_frames(frames, output_path):
    '''
    Given an array of frames extracted from a video, write these frames to an output directory.
    '''
    if output_path != None:
        try:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
        except OSError:
            print ('Error: Creating directory of' + output_path)
    
    print ('Writing frames to ' + output_path)
    currentFrame = 0
    for frame in tqdm(frames):
        # Saves image of the current frame in jpg file
        name = output_path + 'frame' + str(currentFrame) + '.jpg'
        # Write frame to directory 
        cv2.imwrite(name, frame)
        # To stop duplicate images
        currentFrame += 1
    print ("Done!")