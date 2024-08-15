import cv2 
import matplotlib.pyplot as plt
import os
import sys
import torch
import random
from utils.utils_vis.vis import visualize_sample_training, indices_to_rgb
from datetime import datetime      
import json
import wandb
from torch.utils.data import DataLoader
from os import path

def create_video(image_folder, video_name, fps):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # Sort the images by number

    if not images:
        print("No images found in the directory.")
        return

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video_path = os.path.join(image_folder, video_name)
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
        
def plot_metrics(segmentation_loss_values, average_flow_loss_values, total_loss_values, segmentation_accuracy_values, mae_of_values,save_dir):
    print("\n-------------------Plotting Started--------------------------------------\n")

    # Plot the segmentation loss values
    plt.figure(figsize=(10, 5))
    plt.plot(segmentation_loss_values, label='Segmentation Loss')
    plt.plot(average_flow_loss_values, label='Average Flow Loss')
    plt.plot(total_loss_values, label='Total Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss values')
    plt.savefig(f'{save_dir}/figure_loss_values_.png')

    # Plot the segmentation accuracy values
    plt.figure(figsize=(10, 5))
    plt.plot(segmentation_accuracy_values, label='Segmentation Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Segmentation Accuracy values')
    plt.savefig(f'{save_dir}/figure_segmentation_accuracy_.png')

    # Plot the Mean Absolute Error values
    plt.figure(figsize=(10, 5))
    plt.plot(mae_of_values, label='Mean Absolute Error for OF')
    plt.xlabel('Iteration')
    plt.ylabel('MAE')
    plt.legend()
    plt.title('Mean Absolute Error values for OF')
    plt.savefig(f'{save_dir}/figure_MAE_.png')



# Function to handle termination signals
def signal_handler(sig, frame,model):
    print('You pressed Ctrl+C or killed the process!')
    print('Saving model...')
    cwd = os.getcwd()
    save_dir = os.path.join(cwd, 'training_model_saving')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'model.pth')
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}.')
    sys.exit(0)

def visualize_random_sample(dataloader, model, epoch, device,save_dir, config):
    with torch.no_grad():
        num_samples = len(dataloader)
        random_sample_index = random.randint(0, num_samples - 1)

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx == random_sample_index:
            print(f"Batch Index: {batch_idx}")
            visualize_sample_training(batch, model, epoch, device, batch_idx,save_dir, config)
            break

def write_values_to_file(values, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        for item in values:
            f.write("%s\n" % item)

def create_directories_and_names(config,data_name,date, number):
    cwd = os.getcwd()
    now_str = datetime.now().strftime("%Y%m%d_%H%M")
    folder_day_name = f"{data_name}_test_{number}_{now_str}"
    main_folder_name = f"OUTPUT_SEGFLOW_master_thesis/model_saved_output_figures/test_{date}"
    
    full_folder_path = os.path.join(cwd, main_folder_name)
    os.makedirs(full_folder_path, exist_ok=True)
    '''
    current_working_directory
    │
    └───OUTPUT_SEGFLOW_master_thesis
        │
        └───model_saved_output_figures
            │
            └───test_{date}
    '''

    data_name = f'{data_name}'
    #save_config_as_txt(config, main_folder_name,full_folder_path)

    return create_file_names_and_dirs(main_folder_name, data_name, cwd, folder_day_name,config)

def create_file_names_and_dirs(main_folder_name, data_name,cwd, folder_day_name,config):
    # name of the directories
    validation_name= f"VALIDATION_{data_name}"
    training_name = f"TRAINING_{data_name}"
    model_name = f"MODEL_{data_name}"
    plot_name = f"PLOTS_{data_name}"
    model_metrics_values = f"METRICS_{data_name}"
    # paths of the directories
    save_dir_training = f'{main_folder_name}/{folder_day_name}/{training_name}'
    save_dir_validation = f'{main_folder_name}/{folder_day_name}/{validation_name}'
    model_dir_training = f'{main_folder_name}/{folder_day_name}/{model_name}'
    plot_dir = f'{main_folder_name}/{folder_day_name}/{plot_name}'
    model_metrics_dir = f'{main_folder_name}/{folder_day_name}/{model_metrics_values}'

    full_model_dir = os.path.join(cwd, model_dir_training)
    os.makedirs(full_model_dir, exist_ok=True)
    model_file_path = os.path.join(full_model_dir, f'models_{model_name}.pth')


    full_save_dir_training = os.path.join(cwd, save_dir_training)
    os.makedirs(full_save_dir_training, exist_ok=True)

    full_save_dir_validation = os.path.join(cwd, save_dir_validation)
    os.makedirs(full_save_dir_validation, exist_ok=True)

    full_plot_dir = os.path.join(cwd, plot_dir)
    os.makedirs(full_plot_dir, exist_ok=True)

    full_model_metrics_dir = os.path.join(cwd, model_metrics_dir)
    os.makedirs(full_model_metrics_dir, exist_ok=True)

    config_dir = f'{main_folder_name}/{folder_day_name}'
    config_file_path = os.path.join(config_dir, 'configs.txt')
    #os.makedirs(config_file_path, exist_ok=True)
    with open(config_file_path, 'w') as f:
        json.dump(config, f, indent=4)

    return full_model_metrics_dir, model_file_path, full_save_dir_training, full_save_dir_validation, full_plot_dir

def save_config_as_txt(config, main_folder_name, folder_day_name):
    config_dir = f'{main_folder_name}/{folder_day_name}'
    config_file_path = os.path.join(config_dir, 'configs.txt')
    os.makedirs(config_dir, exist_ok=True)
    with open(config_file_path, 'w') as f:
        json.dump(config, f, indent=4)

def get_current_time_str():
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M")



def load_model(model,config, device):
    start_epoch = 0

    if 'model_path' in config and config['model_path']:
        model_path = config['model_path']
        checkpoint = torch.load(model_path)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Loaded model from {model_path} at epoch {start_epoch}")
        else:
            print(f"Loaded entire model from {model_path}")

    print(f"Starting from epoch {start_epoch}")

    model = model.to(device)
    wandb.watch(model)

    return model, start_epoch



def create_dataset_and_dataloader(dataset_class, config, base_path, train_a_transforms, train_torch_transforms, val_torch_transforms):
    ISAAC_dataset_train = dataset_class(
        mask_flow_path=path.join(base_path, 'data', config['train_data']['mask_flow_path'].lstrip('/')),
        segmentation_images_path=path.join(base_path, 'data', config['train_data']['segmentation_images_path'].lstrip('/')),
        segmentation_mask_path=path.join(base_path, 'data', config['train_data']['segmentation_mask_path'].lstrip('/')),
        a_transforms=train_a_transforms,
        torch_transforms=train_torch_transforms,
        start_index=config['train_data']['start_index'],
    )

    ISAAC_dataset_validation = dataset_class(
        mask_flow_path=path.join(base_path, 'data', config['val_data']['mask_flow_path'].lstrip('/')),
        segmentation_images_path=path.join(base_path, 'data', config['val_data']['segmentation_images_path'].lstrip('/')),
        segmentation_mask_path=path.join(base_path, 'data', config['val_data']['segmentation_mask_path'].lstrip('/')),
        a_transforms=None,
        torch_transforms=val_torch_transforms,
        start_index=config['val_data']['start_index']
    )

    dataloader_train = DataLoader(ISAAC_dataset_train, batch_size=config['dataloader']['batch_size'], shuffle=config['dataloader']['shuffle'])
    dataloader_validation = DataLoader(ISAAC_dataset_validation, batch_size=config['dataloader']['batch_size'], shuffle=config['dataloader']['shuffle'])

    return dataloader_train, dataloader_validation