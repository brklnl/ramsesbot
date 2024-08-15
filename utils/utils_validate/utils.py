import torch
import random

import matplotlib.pyplot as plt
import os
import torch
from tqdm import tqdm
import os
from utils.utils_vis.utils import indices_to_rgb
import flow_vis
from utils.utils_vis.vis import class_mapping_combined
from utils.method.utils import visualize_top_tiles


def visualize_results(segmentation_image, segmentation_mask, seg_result, flow_result, processed_image, save_dir_validation, image_counter=1):
    flow_result_color = flow_vis.flow_to_color(flow_result, convert_to_bgr=False)
    segmentation_mask_rgb = indices_to_rgb(segmentation_mask, class_mapping_combined)

    subplot_config = {
        1: {'image': segmentation_image, 'title': 'Segmentation Image'},
        2: {'image': segmentation_mask_rgb, 'title': 'Segmentation Mask'},
        3: {'image': seg_result, 'title': 'Segmentation Prediction'},
        4: {'image': flow_result_color, 'title': 'Optical Flow Prediction'},
        5: {'image': processed_image, 'title': 'Processed Image'},
        6: {'image': flow_result[:,:,0], 'title': 'Flow Result (Channel 1, U)', 'cmap': 'gray'},
        7: {'image': flow_result[:,:,1], 'title': 'Flow Result (Channel 2, V)', 'cmap': 'gray'}
    }

    plt.figure(figsize=(16, 10))
    plt.suptitle(f'Image Number: {image_counter}')  

    for i, config in subplot_config.items():
        plt.subplot(3, 3, i)
        if 'cmap' in config:
            plt.imshow(config['image'], cmap=config['cmap'])
        else:
            plt.imshow(config['image'])
        plt.title(config['title'])

    save_path = os.path.join(save_dir_validation, f"combined_{image_counter}.png")
    plt.savefig(save_path)
    plt.close()  # Close the figure to free memory

    # Increment the counter
    image_counter += 1
    return image_counter





def validate_model(model,dataloader_validation, device, segmentation_criterion, save_dir_validation,config):
    model.eval()  # Set the model to evaluation mode
    segmentation_loss_values = []
    #average_flow_loss_values = []
    #total_loss_values = []
    segmentation_accuracy_values = []
    #mae_of_values = []

    image_counter = 1
    with torch.no_grad():  # Disable gradient calculation
        for batch_idx, batch in tqdm(enumerate(dataloader_validation), total=len(dataloader_validation)):
            # Extract the inputs from the batch
            segmentation_image = batch['segmentation_image'].to(device)
            segmentation_mask = batch['segmentation_mask'].to(device)
            stacked_flow = batch['stacked_flow'].to(device)
            flow_mask = batch['flow_mask'].to(device)

            # Forward pass through the model
            seg_result, flow_result_add = model(segmentation_image, stacked_flow)

            seg_result = torch.softmax(seg_result, dim=1)
            seg_result = torch.argmax(seg_result, dim=1).squeeze().cpu().numpy()

            seg_result_rgb = indices_to_rgb(seg_result, class_mapping_combined)

            # Compute the segmentation accuracy

        #########################################################################
            segmentation_image = segmentation_image.squeeze().permute(1, 2, 0).cpu().numpy()

            segmentation_mask = segmentation_mask.squeeze().cpu().numpy()

            # Define the image size
            image_size = (480, 640)

            # Define the tile size
            tile_size = (image_size[0]//16, image_size[1]//16)  # Adjust the tile size

            # Define the step size
            step_size = tile_size  
            processed_image = visualize_top_tiles(seg_result,
                                                  class_mapping_combined,
                                                  class_scores = config["class_scores"],
                                                  
                                                  image_size = tuple(config["image_size"]),
                                                  tile_size = tuple(config["tile_size"]),
                                                  step_size = step_size
                                                  )

            #############################################################
            # Flow part
            # Flow part
            flow_result = flow_result_add.squeeze().permute(1,2,0).cpu().detach().numpy()

            # Normalize the optical flow vectors

            image_counter = visualize_results(segmentation_image,
                            segmentation_mask,
                            seg_result_rgb, 
                            flow_result,
                            processed_image,
                            save_dir_validation,
                            image_counter
                            )
