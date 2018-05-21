from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 

def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes 
    
    
    """
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    
    #Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
 
    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou

def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

def predict_transform(prediction, input_dim, anchors, num_classes, CUDA=False):
    batch_size = prediction.size(0)
    stride = input_dim // prediction.size(2)
    grid_size = input_dim // stride
    bounding_box_attributes = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bounding_box_attributes * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bounding_box_attributes)
    
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

    # Sigmoid the center_X, center_Y, and confidence
    for i in [0, 1, 4]:
        prediction[:, :, i] = torch.sigmoid(prediction[:, :, i])
    
    # Add the center offsets
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    
    x_y_offset = torch.cat([x_offset, y_offset], 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset

    # Apply the anhors to the dimensions of the bounding box
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()
    
    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)

    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    # apply the sigmoid function to the class scores
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))

    # resize predictions to size of image
    prediction[:,:,:4] *= stride

    return prediction

# def write_results(prediction, confidence, num_classes, nms_conf=0.4):

#     # remove bounding boxes below confidence
#     conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
#     prediction = prediction * conf_mask

#     # create box corners
#     box_corner = prediction.new(prediction.shape)
#     box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
#     box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
#     box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
#     box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
#     prediction[:,:,:4] = box_corner[:,:,:4]

#     batch_size = prediction.size(0)
#     write = False

#     for i in range(batch_size):
#         image_pred = prediction[i]
#         max_conf, max_conf_score = torch.max(image_pred[:, 5:5 + num_classes], 1)
#         max_conf = max_conf.float().unsqueeze(1)
#         max_conf_score = max_conf_score.float().unsqueeze(1)
#         seq = (image_pred[:, :5], max_conf, max_conf_score)
#         image_pred = torch.cat(seq, 1)

#         non_zero_ind = (torch.nonzero(image_pred[:, 4]))
#         try:
#             image_pred = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)
#         except:
#             continue
        
#         # Since using new pytorch this line is necessary
#         if image_pred.shape[0] == 0:
#             continue
        
#         img_classes = unique(image_pred[:, -1]) # -1 holds the class index

#         for img_cls in img_classes:
#             # perform NMS 
#             #get the detections with one particular class
#             cls_mask = image_pred*(image_pred[:,-1] == img_cls).float().unsqueeze(1)
#             class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
#             image_pred_class = image_pred[class_mask_ind].view(-1,7)

#             #sort the detections such that the entry with the maximum objectness
#             #confidence is at the top
#             conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
#             image_pred_class = image_pred_class[conf_sort_index]
#             idx = image_pred_class.size(0)   #Number of detections

#             for i in range(idx):
#                 try:
#                     ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i + 1 : ])
#                 except ValueError:
#                     break
#                 except IndexError:
#                     break
                
#                 iou_mask = (ious < nms_conf).float().unsqueeze(1)
#                 image_pred_class[i + 1:] *= iou_mask

#                 non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
#                 image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)
            
#             batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(i)

#             # Repeat the batch id for as many detections of the class in the image
#             seq = batch_ind, image_pred_class
            
#             if not write:
#                 output = torch.cat(seq, 1)
#                 write = True
#             else:
#                 out = torch.cat(seq, 1)
#                 output = torch.cat((output, out))
            
#     try:
#         return output
#     except:
#         return 0

def write_results(prediction, confidence, num_classes, nms = True, nms_conf = 0.4):
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask
    

    try:
        ind_nz = torch.nonzero(prediction[:,:,4]).transpose(0,1).contiguous()
    except:
        return 0
    
    
    box_a = prediction.new(prediction.shape)
    box_a[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_a[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_a[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_a[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_a[:,:,:4]
    

    
    batch_size = prediction.size(0)
    
    output = prediction.new(1, prediction.size(2) + 1)
    write = False


    for ind in range(batch_size):
        #select the image from the batch
        image_pred = prediction[ind]
        

        
        #Get the class having maximum score, and the index of that class
        #Get rid of num_classes softmax scores 
        #Add the class index and the class score of class having maximum score
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        

        
        #Get rid of the zero entries
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))

        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue
        
        #Get the various classes detected in the image
        img_classes = unique(image_pred_[:,-1])
        
        #WE will do NMS classwise
        for cls in img_classes:
            #get the detections with one particular class
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            

            image_pred_class = image_pred_[class_mask_ind].view(-1,7)

		
        
             #sort the detections such that the entry with the maximum objectness
             #confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)
            
            #if nms has to be done
            if nms:
                #For each detection
                for i in range(idx):
                    #Get the IOUs of all boxes that come after the one we are looking at 
                    #in the loop
                    try:
                        ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                    except ValueError:
                        break
        
                    except IndexError:
                        break
                    
                    #Zero out all the detections that have IoU > treshhold
                    iou_mask = (ious < nms_conf).float().unsqueeze(1)
                    image_pred_class[i+1:] *= iou_mask       
                    
                    #Remove the non-zero entries
                    non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                    image_pred_class = image_pred_class[non_zero_ind].view(-1,7)
                    
                    

            #Concatenate the batch_id of the image to the detection
            #this helps us identify which image does the detection correspond to 
            #We use a linear straucture to hold ALL the detections from the batch
            #the batch_dim is flattened
            #batch is identified by extra batch column
            
            
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class
            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))
    
    return output




