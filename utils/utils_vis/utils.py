import torch.nn as nn
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import wandb
import flow_vis

class FlowToVisTartan:
    def __init__(self, maxF=350.0, n=8, hueMax=179, angShift=0.0):
        self.maxF = maxF
        self.n = n
        self.hueMax = hueMax
        self.angShift = angShift

    def _calculate_angle_distance_from_du_dv(self, du, dv, flagDegree=False):
        a = np.arctan2(dv, du)
        angleShift = np.pi
        if flagDegree:
            a = a / np.pi * 180
            angleShift = 180
        d = np.sqrt(du * du + dv * dv)
        return a, d, angleShift

    def convert(self, flownp, mask=None):
        ang, mag, _ = self._calculate_angle_distance_from_du_dv(flownp[:, :, 0], flownp[:, :, 1], flagDegree=False)
        hsv = np.zeros((ang.shape[0], ang.shape[1], 3), dtype=np.float32)
        am = ang < 0
        ang[am] = ang[am] + np.pi * 2
        hsv[:, :, 0] = np.remainder((ang + self.angShift) / (2*np.pi), 1)
        hsv[:, :, 1] = mag / self.maxF * self.n
        hsv[:, :, 2] = (self.n - hsv[:, :, 1]) / self.n
        hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 1) * self.hueMax
        hsv[:, :, 1:3] = np.clip(hsv[:, :, 1:3], 0, 1) * 255
        hsv = hsv.astype(np.uint8)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        if mask is not None:
            mask = mask > 0
            rgb[mask] = np.array([0, 0 ,0], dtype=np.uint8)
        return rgb

def flow2rgb_flownets(flow_map, max_value):
    #flow_map_np = flow_map.detach().cpu().numpy()
    _, h, w = flow_map.shape
    flow_map[:,(flow_map[0] == 0) & (flow_map[1] == 0)] = float('nan')
    rgb_map = np.ones((3,h,w)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map / max_value
    else:
        normalized_flow_map = flow_map / (np.abs(flow_map).max())
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    return rgb_map.clip(0,1)

def _calculate_angle_distance_from_du_dv(du, dv, flagDegree=False):
    a = np.arctan2( dv, du )

    angleShift = np.pi

    if ( True == flagDegree ):
        a = a / np.pi * 180
        angleShift = 180
        # print("Convert angle from radian to degree as demanded by the input file.")

    d = np.sqrt( du * du + dv * dv )

    return a, d, angleShift

def flow2vis(flownp, maxF=500.0, n=8, mask=None, hueMax=179, angShift=0.0): 

    ang, mag, _ = _calculate_angle_distance_from_du_dv( flownp[:, :, 0], flownp[:, :, 1], flagDegree=False )

    # Use Hue, Saturation, Value colour model 
    hsv = np.zeros( ( ang.shape[0], ang.shape[1], 3 ) , dtype=np.float32)

    am = ang < 0
    ang[am] = ang[am] + np.pi * 2

    hsv[ :, :, 0 ] = np.remainder( ( ang + angShift ) / (2*np.pi), 1 )
    hsv[ :, :, 1 ] = mag / maxF * n
    hsv[ :, :, 2 ] = (n - hsv[:, :, 1])/n

    hsv[:, :, 0] = np.clip( hsv[:, :, 0], 0, 1 ) * hueMax
    hsv[:, :, 1:3] = np.clip( hsv[:, :, 1:3], 0, 1 ) * 255
    hsv = hsv.astype(np.uint8)

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    if ( mask is not None ):
        mask = mask > 0
        rgb[mask] = np.array([0, 0 ,0], dtype=np.uint8)

    return rgb

def seg2vis_mask_class_mapping(segnp, class_mapping):
    segvis = np.zeros(segnp.shape+(3,), dtype=np.uint8)

    for k, color in class_mapping.items():
        mask = segnp==k
        if np.sum(mask)>0:
            segvis[mask,:] = color
    return segvis


def indices_to_rgb(indices, class_mapping):
    # Define the class mapping


    rgb_image = np.zeros(indices.shape+(3,), dtype=np.uint8)

    for k in class_mapping.keys():
        mask = indices==k
        if np.sum(mask)>0:
            rgb_image[mask] = class_mapping[k]
    return rgb_image



    

def flow2rgb(flow_map, max_value):
    flow_map_np = flow_map.copy()
    h, w, _ = flow_map_np.shape
    
    # Create a boolean mask for zero flow values
    zero_flow_mask = (flow_map_np[..., 0] == 0) & (flow_map_np[..., 1] == 0)
    
    # Apply the mask to set zero flow values to NaN
    flow_map_np[zero_flow_mask] = float('nan')

    rgb_map = np.ones((h, w, 3)).astype(np.float32)

    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else:
        normalized_flow_map = flow_map_np / np.abs(flow_map_np).max()

    # Perform channel-wise addition
    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]

    return rgb_map.clip(0, 1)


def visualize_optical_flow(flow_result_add_scaled):
    # Calculate the magnitude and angle of the 2D vectors
    # Ensure values are within the expected range for cv2.cartToPolar
    #flow_result_add_scaled[..., 0] = np.clip(flow_result_add_scaled[..., 0], -1.0, 1.0)
    #flow_result_add_scaled[..., 1] = np.clip(flow_result_add_scaled[..., 1], -1.0, 1.0)
    #flow_result_add_scaled = flow_result_add_scaled.astype(np.float32)

    # Scale the values to the range [-pi, pi]
    #flow_result_add_scaled[..., 0] *= np.pi
    #flow_result_add_scaled[..., 1] *= np.pi
    magnitude, angle = cv2.cartToPolar(flow_result_add_scaled[..., 0], flow_result_add_scaled[..., 1])

    # Set the hue according to the angle (direction)
    hue = angle * 180 / np.pi / 2

    # Normalize the magnitude for better visualization
    value = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)

    # Convert HSV to RGB for display in Matplotlib
    rgb_image = plt.cm.hsv(hue / 180.0)
    rgb_image[..., 2] = value

    return rgb_image


def visualize_normal_way_sample(batch, model, device):
    x = batch['segmentation_image'].to(device)
    y = batch['segmentation_mask'].to(device)  # assuming 'segmentation_mask' is the key for the target mask
    stacked_flow = batch['stacked_flow'].to(device)
    fig , ax =  plt.subplots(1, 3, figsize=(18, 18))
    softmax = nn.Softmax(dim=1)
    seg_result, flow_result_add = model(x, stacked_flow)
    preds = torch.argmax(softmax(seg_result),axis=1).to('cpu')

    # Select the sample at the random index
    img = np.transpose(np.array(x[0,:,:,:].to('cpu')),(1,2,0))
    pred = np.array(preds[0,:,:])
    mask = np.array(y[0,:,:])

    ax[0].set_title('Image')
    ax[1].set_title('Prediction')
    ax[2].set_title('Mask')
    ax[0].axis("off")
    ax[1].axis("off")
    ax[2].axis("off")
    ax[0].imshow(img)
    ax[1].imshow(pred)
    ax[2].imshow(mask)
    #plt.show()