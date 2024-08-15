import torch
import numpy as np
import matplotlib.pyplot as plt
import os
#######################################
import torch
from PIL import Image
import flow_vis
from utils.utils_vis.utils import indices_to_rgb
from tqdm import tqdm
from method.landing_site_detection_v2_0 import ImageProcessor
from matplotlib import patches
import cv2

def process_segmentation_output(seg_result, class_mapping):
    seg_result = torch.softmax(seg_result, dim=1)
    seg_result = torch.argmax(seg_result, dim=1).squeeze().cpu().numpy()
    seg_result_rgb = indices_to_rgb(seg_result, class_mapping)
    return seg_result_rgb, seg_result

def process_flow_output(flow_result):
    flow_result = flow_result.squeeze().permute(1, 2, 0).cpu().detach().numpy()
    converted_flow_result = -flow_result
    flow_result_rgb = flow_vis.flow_to_color(converted_flow_result, convert_to_bgr=True)
    
    return flow_result_rgb, flow_result

def plot_image(ax, image, title, cmap=None, alpha=None):
    ax.imshow(image, cmap=cmap, alpha=alpha)
    ax.set_title(title)
    ax.axis('off')

def plot_heatmap(ax, data, title):
    plot_image(ax, data, title, cmap='hot')



def plot_landing_sites_new(ax, seg_result, s, original_image, tile_size, step_size, alpha=0.4):
    # Display the original image
    ax.imshow(original_image)
    
    # Overlay the segmentation result with transparency
    ax.imshow(seg_result, alpha=alpha)
    
    # Find the indices of the top 5 scores in the s array
    top_scores_indices = np.argpartition(s.flatten(), -5)[-5:]
    top_scores = s.flatten()[top_scores_indices]
    sorted_indices = top_scores_indices[np.argsort(-top_scores)]
    
    # Colors for the top 5 tiles
    colors = ['g'] + ['r'] * 4
    
    for rank, index in enumerate(sorted_indices):
        # Convert the index to 2D coordinates (row, column) in the grid of tiles
        score_2d_index = np.unravel_index(index, s.shape)
        # Calculate the original image coordinates of the tile
        top_left_y = score_2d_index[0] * step_size
        top_left_x = score_2d_index[1] * step_size
        bottom_right_y = top_left_y + tile_size[0]
        bottom_right_x = top_left_x + tile_size[1]
        
        # Draw a rectangle around the tile with the corresponding color
        rect = patches.Rectangle((top_left_x, top_left_y), tile_size[1], tile_size[0], linewidth=2, edgecolor=colors[rank], facecolor='none')
        ax.add_patch(rect)
        
        # Print the coordinates
        print(f"{['First', 'Second', 'Third', 'Fourth', 'Fifth'][rank]} best tile's coordinates: Top Left ({top_left_y}, {top_left_x}), Bottom Right ({bottom_right_y}, {bottom_right_x})")
def plot_best_landing_site(ax, s, title=None):
    # Plot the heatmap of s
    ax.imshow(s, cmap='hot')
    if title:
        ax.set_title(title)
    ax.axis('off')

    # Find the maximum score and its index
    max_score_index = np.unravel_index(np.argmax(s), s.shape)
    max_score = s[max_score_index]

    # Calculate the dimensions of each tile directly from s's shape
    tile_height = 1  # Each tile's height is 1 unit in the context of the heatmap
    tile_width = 1   # Each tile's width is 1 unit in the context of the heatmap

    # Calculate the top-left corner of the tile
    top_left = (max_score_index[1] * tile_width, max_score_index[0] * tile_height)

    # Highlight the tile with a rectangle
    rect = patches.Rectangle(top_left, tile_width, tile_height, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    # Annotate the rectangle with the score
    ax.text(top_left[0], top_left[1]-0.1, f'Max Score: {max_score:.2f}', color='r', fontsize=9, ha='left')



def plot_optical_flow(ax, flow, title):
    flow_rgb = flow_vis.flow_to_color(flow, convert_to_bgr=True)
    ax.imshow(flow_rgb)
    ax.set_title(title)
    ax.axis('off')

def save_visualization(image, seg_result_rgb, flow_result_rgb, scores, top_tiles, save_dir, counter, s, sc_scaled, so_resized_scaled, config, flow_gt):
    fig, axs = plt.subplots(4, 3, figsize=(25, 20))
    tile_size = config['tile_size_sg']
    step_size = config['step_size_sg']
    
    # Plot original image, segmentation output, and optical flow output
    plot_image(axs[0, 0], image, 'Original Image')
    plot_image(axs[0, 1], seg_result_rgb, 'Segmentation Output')
    plot_image(axs[0, 2], flow_result_rgb, 'Optical Flow Output')

    plot_heatmap(axs[1, 0], sc_scaled, 'Heatmap of SG')
    plot_heatmap(axs[1, 1], so_resized_scaled, 'Heatmap of SO')
    plot_best_landing_site(axs[1, 2], s, 'Heatmap of Mixed Score')

    # Plot overlay of original and segmentation output
    plot_image(axs[2, 0], image, 'Original Image')
    plot_image(axs[2, 0], seg_result_rgb, 'Segmentation Overlay', alpha=0.5)

    # Plot overlay of original and optical flow output
    plot_image(axs[2, 1], image, 'Original Image')
    plot_image(axs[2, 1], flow_result_rgb, 'Optical Flow Overlay', alpha=0.6)

    plot_landing_sites_new(axs[2, 2], seg_result_rgb, s, image, tile_size, step_size)

    # Plot optical flow ground truth
    plot_optical_flow(axs[3, 1], flow_gt, 'Optical Flow Ground Truth')

    # Plot optical flow ground truth heatmap
    magnitude = np.sqrt(flow_gt[..., 0]**2 + flow_gt[..., 1]**2)
    plot_heatmap(axs[3, 0], magnitude, 'Heatmap of Optical Flow GT')
  # Plot optical flow ground truth overlay
    plot_image(axs[3, 2], image, 'Original Image')
    plot_image(axs[3, 2], flow_vis.flow_to_color(flow_gt, convert_to_bgr=True), 'Optical Flow GT Overlay', alpha=0.4)

    axs[3, 2].axis('off')

    plt.savefig(os.path.join(save_dir, f'output_{counter}.png'))
    plt.close(fig)

def test_model_and_lsd(model, dataloader, device, save_dir, config):
    model.eval()
    global_counter = 0
    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        image = batch['image'].to(device)
        stacked_images = batch['stacked_images'].to(device)
        seg_result, flow_result = model(image, stacked_images)
        
        class_mapping = {int(k): v for k, v in config['class_mapping'].items()}
        scores = config['class_scores']
        
        seg_result_rgb, seg_result = process_segmentation_output(seg_result, class_mapping)
        flow_result_rgb, flow_result = process_flow_output(flow_result)
        
        original_image = image[0].cpu().numpy()
        original_image = np.transpose(original_image, (1, 2, 0))
        original_image_pil = Image.fromarray((original_image * 255).astype(np.uint8)).resize((640, 480))
        original_image = np.array(original_image_pil)
        
        # Create an instance of ImageProcessor
        processor = ImageProcessor(config['class_scores'], class_mapping, original_image, 
                                   tuple(config['tile_size_sg']), config['step_size_sg'], 
                                   config['tile_size_of'])
        
        # Call the process_images method
        safe_tiles, top_tiles, sc_scaled, s, so_resized_scaled = processor.process_images(flow_result, seg_result)
        
        # Compute optical flow ground truth using Farneback method
        prev_image = stacked_images[0, :3, :, :].cpu().numpy().transpose(1, 2, 0)
        next_image = stacked_images[0, 3:, :, :].cpu().numpy().transpose(1, 2, 0)
        prev_gray = cv2.cvtColor((prev_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        next_gray = cv2.cvtColor((next_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        flow_gt = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
         
        # Use a default value for 'scores' if it's not found in the config dictionary
        if image.is_cuda:
            image = image.squeeze().permute(1, 2, 0).cpu().numpy()
        else:
            image = image.numpy()
        
        save_visualization(image, seg_result_rgb, flow_result_rgb, scores, top_tiles, save_dir, global_counter, s, sc_scaled, so_resized_scaled, config, flow_gt)
        global_counter += 1
def save_output(model, dataloader, device, save_dir, config):
    model.eval()
    global_counter = 0
    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        image = batch['image'].to(device)
        stacked_images = batch['stacked_images'].to(device)
        seg_result, flow_result = model(image, stacked_images)
        class_mapping = {int(k): v for k, v in config['class_mapping'].items()}

        # Process the segmentation and flow results
        _, seg_result = process_segmentation_output(seg_result, class_mapping)
        _, flow_result = process_flow_output(flow_result)
        # Print the shape of the processed outputs
        print(f'Segmentation result shape: {seg_result.shape}')
        print(f'Flow result shape: {flow_result.shape}')
        # Ensure the save directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save processed segmentation and optical flow results
        np.save(os.path.join(save_dir, f'seg_result_rgb_{global_counter}.npy'), seg_result)
        np.save(os.path.join(save_dir, f'flow_result_rgb_{global_counter}.npy'), flow_result)
        print(f'Saved seg_result_rgb_{global_counter}.npy and flow_result_rgb_{global_counter}.npy')

        global_counter += 1