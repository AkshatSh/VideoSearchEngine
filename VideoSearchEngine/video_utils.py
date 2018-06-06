import cv2
import numpy as np
import os
from tqdm import tqdm
from skimage.measure import compare_ssim as ssim
import pytorch_ssim
import torch
from torch.autograd import Variable

'''
This util file is for anything related to video processing that can be factored out into here
'''

def get_frames_clusters_from_video(video_path, cluster_size=425):
    '''
    Given a video path, read the video, store every cluster_size frames in an array add it a list
    and return it.
    '''
    # Playing video from file:
    frameClustersArray = []
    cluster = []
    cap = cv2.VideoCapture(video_path)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Break if no image is returned (have reached end of video)
        if frame is None or ret is False:
            break
        # Add frame to array
        cluster.append(frame)
        # If cluster is cluster size break it off
        if len(cluster) == cluster_size:
            frameClustersArray.append(list(cluster))
            cluster.clear()

    # Append any residual frames
    if len(cluster) > 0:
        frameClustersArray.append(list(cluster))
        cluster.clear()  

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    return frameClustersArray

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


def group_semantic_frames(frames, threshold=None):
    '''
    Given an array of frames extracted from a video, break these into subarrays of semantically similiar frames.
    For now use the Structural Similarity Index and once it reaches a certain threshold break off. Return an array of
    these sub arrays.
    '''
    frame_clusters = []
    group = []
    print("Grouping semantic frames")
    for frame in tqdm(frames):
        if len(group) == 0:
            group.append(frame)
        else:
            # compute structural similarity index between current image and oldest image in the frame group
            s = 0.0
            if torch.cuda.is_available():
                threshold = .28
                img1 = torch.from_numpy(np.rollaxis(group[0], 2)).float().unsqueeze(0)/255.0
                img2 = torch.from_numpy(np.rollaxis(frame, 2)).float().unsqueeze(0)/255.0
                img1 = img1.cuda()
                img2 = img2.cuda()
                img1 = Variable( img1,  requires_grad=False)
                img2 = Variable( img2, requires_grad = True)
                s = pytorch_ssim.ssim(img1, img2)
                s = s.cpu().data.numpy()
            else:
                threshold = 0.35
                s = ssim(group[0], frame, multichannel=True)
            if s < threshold:
                frame_clusters.append(list(group))
                group.clear()
            # TODO: If we don't append the frame each time we only get the salient images which reduces number of frames
            group.append(frame)
    
    if len(group) > 0:
        frame_clusters.append(list(group))
    return frame_clusters
