from PIL import Image
import torch
import random

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
#from utils import iou, dice_coefficient

from utils.vis import class_mapping
#######################################
from dataset import TartanAir_1
from torchvision import transforms
from torch.utils.data import DataLoader
import wandb
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
from torchvision import transforms
from PIL import Image
from models.model import SilkiMaus
from utils.vis import visualize_sample_validation
from utils.utils_vis import FlowToVisTartan, flow2rgb_flownets, flow2rgb_flownets, _calculate_angle_distance_from_du_dv, flow2vis, seg2vis_mask_class_mapping, indices_to_rgb
import flow_vis

# Load the model
import os
import stat


class_mapping = {
    0: [140, 255, 25],  # color 1
    1: [255, 197, 25],  # color 2
    2: [140, 25, 255],  # color 3
    3: [25, 82, 255],  # color 4
    4: [25, 255, 82],  # color 5
    5: [255, 25, 197]  # color 6
}


# Load the saved model
file_path = "/home/ist-sim/burak/SEGFLOW_master_thesis/OUTPUT_SEGFLOW_master_thesis/model_saved_output_figures/test_10.06/ISAAC_test_1.0_of_20240610_2239_3k/TRAINING_ISAAC_first_test_Adam_0.0001_epoch_200_weight_decay0.01/model_best_at_60_epoch.pth"

# Check if the file exists
if os.path.exists(file_path):
    print("File exists")
else:
    print("File does not exist")

# Check if the file is readable
if os.access(file_path, os.R_OK):
    print("File is accessible")
else:
    print("File is not accessible")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = SilkiMaus()

# Load the state dictionary
state_dict = torch.load(file_path)  # Corrected here

# Update the model's state dictionary
model.load_state_dict(state_dict)

# Move the model to the device
model = model.to(device)

# Load the first image
image_path1 = "/home/ist-sim/burak/SEGFLOW_master_thesis/data/ISAAC_ENV_2_3k_new/val/rgb/rgb_000850.png"
image1 = Image.open(image_path1).convert("RGB")

# Load the second image
image_path2 = "/home/ist-sim/burak/SEGFLOW_master_thesis/data/ISAAC_ENV_2_3k_new/val/rgb/rgb_000851.png"
image2 = Image.open(image_path2).convert("RGB")
# Get the original size of the image
original_size = image1.size

# Calculate the new size
new_size = (original_size[0] // 2, original_size[1] // 2)

# Apply the same transformations as you did for your validation data
transform = transforms.Compose([
    transforms.Resize((640,480)),  # Add this line
    transforms.ToTensor()
])

# Apply the transformations to the images
image1 = transform(image1)
image2 = transform(image2)
print(image1.shape)

# Stack the images along a new dimension
images = torch.cat([image1, image2], dim=0)

# Move the images to the device
image1 = image1.to(device)
images = images.to(device)
# Add a batch dimension to the images
image1 = image1.unsqueeze(0)
images = images.unsqueeze(0)

# Now, image1 and images should have a batch size of 1
print(image1.shape)
print(images.shape)


import os

# Define the directory to save the results
output_dir = "/home/ist-sim/burak/SEGFLOW_master_thesis/OUTPUT_SEGFLOW_master_thesis/VAL_OUTPUT/ISAAC/single_image_test_output"

# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

import matplotlib.pyplot as plt

# Pass the images through the model
with torch.no_grad():
    model.eval()
    seg_result, flow_result_add = model(image1,images)

    # Process the output as you did before
    seg_result = torch.softmax(seg_result, dim=1)
    seg_result = torch.argmax(seg_result, dim=1).squeeze().cpu().numpy()
    seg_result = np.transpose(seg_result)
    seg_result_rgb = indices_to_rgb(seg_result, class_mapping)

    # Save the segmentation result
    plt.imsave(os.path.join(output_dir, 'seg_result_rgb2.png'), seg_result_rgb)
    np.save(os.path.join(output_dir, 'seg_result.npy'), seg_result)

    # Convert the optical flow result to a color image
    flow_result_add = flow_result_add.squeeze().permute(1,2,0).cpu().numpy()
    flow_result_add = np.transpose(flow_result_add, (1,0,2))
    flow_result_rgb = flow_vis.flow_to_color(flow_result_add, convert_to_bgr=False)

    # Save the optical flow result
    plt.imsave(os.path.join(output_dir, 'flow_result_rgb_2.png'), flow_result_rgb)

    # Open the original image as a numpy array
    original_image_np = plt.imread(image_path1)

    # Open the segmentation result as a numpy array
    seg_result_np = plt.imread(os.path.join(output_dir, 'seg_result_rgb2.png'))

    # Create a new figure
    plt.figure()

    # Display the original image
    plt.imshow(original_image_np)

    # Overlay the segmentation result with an alpha of 0.5
    plt.imshow(seg_result_np, alpha=0.5)
    plt.axis('off')

    # Save the overlay
    plt.savefig(os.path.join(output_dir, 'overlay.png'))



    # # After saving the segmentation result, open it as a PIL image
    # seg_result_pil = Image.open(os.path.join(output_dir, 'seg_result_rgb.png')).convert("RGBA")

    # # Open the original image as a PIL image, resize it to match the segmentation result, and convert it to the same mode
    # original_image_pil = Image.open(image_path1).convert("RGBA").resize(seg_result_pil.size)

    # # Blend the original image and the segmentation result
    # overlay = Image.blend(original_image_pil, seg_result_pil, alpha=0.5)

    # # Save the overlay
    # overlay.save(os.path.join(output_dir, 'overlay.png'))