import torch.nn as nn
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from utils.utils_training.utils_losses.of_loss import one_scale
import wandb
import flow_vis
from utils.utils_vis.utils import FlowToVisTartan, flow2rgb_flownets, flow2rgb_flownets, _calculate_angle_distance_from_du_dv, flow2vis, seg2vis_mask_class_mapping, indices_to_rgb
#from utils import denormalize    

# standard_color_mapping = {
#     'BACKGROUND': '(0, 0, 0, 0)',
#     'ground': '(140, 255, 25, 255)', # class 1
#     'gravel': '(140, 25, 255, 255)', # class 2
#     'flat_rock': '(25, 255, 82, 255)', # class 3
#     'rock1': '(25, 82, 255, 255)', # class 4
#     'rock2': '(255, 197, 25, 255)', # class 5
#     'big_rock': '(255, 25, 197, 255)', # class 6
#     'UNLABELLED': '(0, 0, 0, 255)'
# }
# class_mapping = {
#     0: [140, 255, 25],  # ground
#     1: [140, 25, 255],  # gravel
#     # 2: [25, 255, 82],  # flat_rock
#     # 3: [25, 82, 255], # rock1
#     # 4: [255, 197, 25], # rock2
#     # 5: [255, 25, 197]    # big_rock
# }

# class_mapping_VAL = {
#     0: [140, 255, 25],  # ground
#     1: [140, 25, 255],  # gravel
#     2: [25, 255, 82],  # flat_rock
#     3: [25, 82, 255], # rock1
#     4: [255, 197, 25], # rock2
#     5: [255, 25, 197]    # big_rock
# }

# class_mapping = {
#     0: [140, 255, 25],  # color 1
#     1: [255, 197, 25],  # color 2
#     2: [140, 25, 255],  # color 3
#     3: [25, 82, 255],  # color 4
#     4: [25, 255, 82],  # color 5
#     5: [255, 25, 197]  # color 6
# }
# class_mapping_env2_VAL = {
#     0: [140, 255, 25],  # color 1
#     1: [255, 197, 25],  # color 2
#     2: [140, 25, 255],  # color 3
#     3: [25, 82, 255],  # color 4
#     4: [25, 255, 82],  # color 5
#     5: [255, 25, 197]  # color 6
# }


# class_mapping_combined = {
#     0: [25, 255, 82], # bushes, light green
#     1: [255, 165, 0], # mid_rock, orange
#     2: [169, 169, 169], # gravel, dark gray
#     3: [139, 69, 19], #ground, brown
#     4: [255, 0, 0], # big rock, red
#     5: [0, 0, 255], # flat rock, blue
# }

def load_sample(sample, device):
    """Load a sample into the specified device."""
    return {key: value.to(device) for key, value in sample.items()}

def compute_segmentation_accuracy(seg_result, segmentation_mask):
    """Compute the segmentation accuracy."""
    correct_pixels = (seg_result == segmentation_mask.cpu().numpy()).sum()
    total_pixels = segmentation_mask.nelement()
    return correct_pixels / total_pixels

def compute_flow_loss(flow_result, flow_mask):
    """Compute the flow loss."""
    flow_loss = one_scale(flow_result, flow_mask)
    num_elements = flow_result.numel()
    return flow_loss.item() / num_elements

def compute_mean_absolute_error(flow_result, flow_mask):
    """Compute the mean absolute error."""
    absolute_difference = torch.abs(flow_result - flow_mask)
    return absolute_difference.mean().item()

def load_mask(mask_dir, sample_idx):
    """Load the mask file."""
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.npy')], key=lambda f: int(f.split('_')[2].split('.')[0]))
    mask_file = mask_files[sample_idx]
    return np.load(os.path.join(mask_dir, mask_file))

def load_and_prepare_sample(sample, device):
    sample = load_sample(sample, device)
    segmentation_image = sample['segmentation_image']
    segmentation_mask = sample['segmentation_mask']
    stacked_flow = sample['stacked_flow']
    flow_mask = sample['flow_mask']
    return segmentation_image, segmentation_mask, stacked_flow, flow_mask

def process_segmentation_and_flow(model, segmentation_image, stacked_flow, class_mapping):
    seg_result, flow_result = model(segmentation_image, stacked_flow)
    seg_result = torch.softmax(seg_result, dim=1)
    seg_result = torch.argmax(seg_result, dim=1).squeeze().cpu().numpy()
    seg_result_rgb = indices_to_rgb(seg_result, class_mapping)
    return seg_result, flow_result, seg_result_rgb

def compute_accuracy_and_mae(seg_result, segmentation_mask, flow_result, flow_mask):
    segmentation_accuracy = compute_segmentation_accuracy(seg_result, segmentation_mask)
    mae = compute_mean_absolute_error(flow_result, flow_mask)
    return segmentation_accuracy, mae

def prepare_for_logging(segmentation_image, segmentation_mask, stacked_flow, flow_mask, flow_result, class_mapping):
    segmentation_image = segmentation_image.squeeze().permute(1, 2, 0).cpu().numpy()
    segmentation_mask = segmentation_mask.squeeze().cpu().numpy()
    stacked_flow = stacked_flow.permute(0, 2, 3, 1).squeeze().cpu().numpy()
    flow_result = flow_result.squeeze().permute(1, 2, 0).cpu().detach().numpy()
    flow_mask = flow_mask.squeeze().permute(1, 2, 0).cpu().numpy()
    mask_vis = flow_vis.flow_to_color(flow_mask[:, :, :2], convert_to_bgr=False)
    return segmentation_image, segmentation_mask, stacked_flow, flow_result, flow_mask, mask_vis

def prepare_visualizations(segmentation_image, segmentation_mask, stacked_flow, flow_mask, flow_result, class_mapping):
    segmentation_image = segmentation_image.squeeze().permute(1, 2, 0).cpu().numpy()
    segmentation_mask = segmentation_mask.squeeze().cpu().numpy()
    stacked_flow = stacked_flow.permute(0, 2, 3, 1).squeeze().cpu().numpy()
    flow_result = flow_result.squeeze().permute(1, 2, 0).cpu().detach().numpy()
    flow_mask = flow_mask.squeeze().permute(1, 2, 0).cpu().numpy()
    mask_vis = flow_vis.flow_to_color(flow_mask[:, :, :2], convert_to_bgr=False)
    return segmentation_image, segmentation_mask, stacked_flow, flow_result, flow_mask, mask_vis

def log_results(segmentation_image, segmentation_mask, seg_result, mask_vis, flow_result, segmentation_accuracy, mae, class_mapping, mode="TR"):
    prefix = "VAL" if mode == "evaluation" else "TR"
    wandb.log({
        f"{prefix} Segmentation Image": wandb.Image(segmentation_image),
        f"{prefix} Segmentation Mask": wandb.Image(indices_to_rgb(segmentation_mask, class_mapping)),
        f"{prefix} Segmentation Result (Accuracy: {segmentation_accuracy:.2f})": wandb.Image(indices_to_rgb(seg_result, class_mapping)),
        f"{prefix} Optical Flow with Mask": wandb.Image(mask_vis),
        f"{prefix} Flow Result (MAE: {mae:.2f})": wandb.Image(flow_vis.flow_to_color(flow_result, convert_to_bgr=False)),
        f"{prefix} Flow Result U": wandb.Image(flow_result[:, :, 0]),
        f"{prefix} Flow Result V": wandb.Image(flow_result[:, :, 1])
    })

def visualize_sample_training(sample, model, epoch, device, sample_idx, save_dir, config):
    """Visualize the training of a sample."""
    segmentation_image, segmentation_mask, stacked_flow, flow_mask = load_and_prepare_sample(sample, device)


    seg_result, flow_result = process_segmentation_and_flow(model, segmentation_image, stacked_flow)


    segmentation_accuracy, mae = compute_accuracy_and_mae(seg_result, segmentation_mask, flow_result, flow_mask)

    class_mapping = {int(k): v for k, v in config['class_mapping_combined'].items()}


    segmentation_image, segmentation_mask, stacked_flow, flow_result, flow_mask, mask_vis = prepare_visualizations(segmentation_image, segmentation_mask, stacked_flow, flow_mask, flow_result, class_mapping)


    #########################################################################
    '''Plotting and Log Part'''
    ######################################################################
    log_results(segmentation_image, segmentation_mask, seg_result, mask_vis, flow_result, segmentation_accuracy, mae, class_mapping, mode='TR')
    ######################################################################
    # # Define a dictionary to hold your subplot configurations
    # subplot_config = {
    #     1: {'image': segmentation_image, 'title': 'Segmentation Image'},
    #     2: {'image': indices_to_rgb(segmentation_mask, class_mapping), 'title': 'Segmentation Mask'},
    #     3: {'image': indices_to_rgb(seg_result, class_mapping), 'title': f'Segmentation Result (Accuracy: {segmentation_accuracy:.2f})'},
    #     4: {'image': stacked_flow[:,:,:3], 'title': 'Stacked Flow First Image'},
    #     5: {'image': stacked_flow[:,:,3:6], 'title': 'Stacked Flow Second Image'},
    #     6: {'image': flow_vis.flow_to_color(flow_result, convert_to_bgr=False), 'title': 'Flow Result'},
    #     7: {'image': mask_vis, 'title': 'Optical Flow with Mask'},
    #     8: {'image': flow_result[:,:,0], 'title': f'Flow Result (Channel 1, U) (MAE: {mae:.2f})', 'cmap': 'gray'},
    #     9: {'image': flow_result[:,:,1], 'title': f'Flow Result (Channel 2, V) (MAE: {mae:.2f})', 'cmap': 'gray'}
    # }

    # plt.figure(figsize=(16, 10))
    # plt.suptitle(f'Image Number: {sample_idx+1}')  

    # # Loop through the dictionary and create subplots
    # for i, config in subplot_config.items():
    #     plt.subplot(3, 3, i)
    #     if 'cmap' in config:
    #         plt.imshow(config['image'], cmap=config['cmap'])
    #     else:
    #         plt.imshow(config['image'])
    #     plt.title(config['title'])

    # plt.savefig(f'{save_dir}/figure_epoch_{epoch}.png')
#####################################################################################################################################


######################################################################################################################################
def visualize_sample_validation(validation_loader, model, epoch, device, sample_idx, save_dir, config):
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(validation_loader):
            if i >= 5:  # Stop after 5 samples
                break

            segmentation_image, segmentation_mask, stacked_flow, flow_mask = load_and_prepare_sample(sample, device)
            class_mapping = {int(k): v for k, v in config['class_mapping_combined'].items()}
            seg_result, flow_result, seg_result_rgb = process_segmentation_and_flow(model, segmentation_image, stacked_flow, class_mapping)
            segmentation_accuracy, mae = compute_accuracy_and_mae(seg_result, segmentation_mask, flow_result, flow_mask)
            segmentation_image, segmentation_mask, stacked_flow, flow_result, flow_mask, mask_vis = prepare_for_logging(segmentation_image, segmentation_mask, stacked_flow, flow_mask, flow_result, class_mapping)
            log_results(segmentation_image, segmentation_mask, seg_result_rgb, mask_vis, flow_result, segmentation_accuracy, mae, class_mapping, mode='VAL')
            # subplot_config = {
            #     1: {'image': segmentation_image, 'title': 'Segmentation Image'},
            #     2: {'image': indices_to_rgb(segmentation_mask, class_mapping), 'title': 'Segmentation Mask'},
            #     3: {'image': seg_result_rgb, 'title': f'Segmentation Result (Accuracy: {segmentation_accuracy:.2f})'},
            #     4: {'image': stacked_flow[:,:,:3], 'title': 'Stacked Flow First Image'},
            #     5: {'image': stacked_flow[:,:,3:6], 'title': 'Stacked Flow Second Image'},
            #     6: {'image': mask_vis, 'title': 'Optical Flow with Mask'},
            #     7: {'image': flow_vis.flow_to_color(flow_result, convert_to_bgr=False), 'title': f'Flow Result (Channel 1) (MAE: {mae:.2f})'}
            # }

            # plt.figure(figsize=(16, 10))
            # plt.suptitle(f'Image Number: {sample_idx+1}')  

            # for i, subplot_config_item in subplot_config.items():
            #     plt.subplot(2, 4, i)
            #     plt.imshow(subplot_config_item['image'])
            #     plt.title(subplot_config_item['title'])

            # if not os.path.exists(save_dir):
            #     os.makedirs(save_dir)
                        
            # plt.savefig(f'{save_dir}/validation_figure_epoch_{epoch}_sample_{sample_idx}.png')



