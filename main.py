import torch.nn as nn
from torchvision import transforms
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
import signal
from functools import partial
import json
######################################
from dataset.dataset import Isaac_Data
from models.model import SilkiMaus
from utils.utils import plot_metrics, signal_handler
from loop.train_new import train_model
from utils.utils_training.utils_losses.of_loss import one_scale
import wandb
from utils.utils import create_file_names_and_dirs, create_directories_and_names
from utils.utils_training.utils_transform.utils import A_transforms, torch_transforms
import os
from utils.utils import load_model
from utils.utils import create_dataset_and_dataloader


#######################################
print(os.getcwd())
import os

base_path = os.getcwd()

with open(os.path.join(base_path, 'master_thesis', 'main_test_2.1_isaac_data', 'configs','main_train_config.json')) as f:
    config = json.load(f)

run = wandb.init(project=config['project'], entity=config['entity'])

torch.cuda.empty_cache()

train_a_transforms = A_transforms()
train_torch_transforms = transforms.Compose([transforms.ToTensor()])

val_torch_transforms = transforms.Compose([transforms.ToTensor()])

dataloader_train, dataloader_validation = create_dataset_and_dataloader(Isaac_Data,
                                                                        config,
                                                                        base_path, 
                                                                        train_a_transforms, 
                                                                        train_torch_transforms, 
                                                                        val_torch_transforms)


############################################################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SilkiMaus(num_classes=config['num_classes'])
model, start_epoch = load_model(model, config, device)

# Check the device of the first parameter
first_param_device = next(model.parameters()).device

print(first_param_device)
###########################################################################################################
# Set the signal handler
# Create a new function that has 'model' as a pre-filled argument
handler_with_model = partial(signal_handler, model=model)
# Set the signal handler
signal.signal(signal.SIGINT, handler_with_model)
signal.signal(signal.SIGTERM, handler_with_model)
###########################################################################################################

optimizer = optim.Adam(model.parameters(), lr=config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'], betas=tuple(config['optimizer']['betas']))

scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config['scheduler']['milestones'], gamma=config['scheduler']['gamma'])

segmentation_criterion = nn.CrossEntropyLoss()
flow_criterion = one_scale

num_epochs = config['num_epochs']

###########################################################################################################


full_model_metrics_dir, model_file_path, full_save_dir_training, \
full_save_dir_validation, full_plot_dir = create_directories_and_names(config,
                                                                       data_name=config['data_name'],
                                                                              date=config['date'],
                                                                                number=config['training_number'])
print(full_save_dir_training)




# Call the train_model function
segmentation_accuracy_values, mae_of_values, segmentation_loss_values, \
average_flow_loss_values, total_loss_values, pixel_difference_values, \
iou_values, dice_coefficient_values = train_model(
    model, 
    dataloader_train,
    dataloader_validation, 
    device,  
    segmentation_criterion, 
    flow_criterion, 
    optimizer, 
    scheduler, 
    full_save_dir_training, 
    full_save_dir_validation,
    full_model_metrics_dir,
    config,
    start_epoch
    )
plot_metrics(segmentation_loss_values, average_flow_loss_values, total_loss_values, segmentation_accuracy_values, mae_of_values,full_plot_dir)

torch.save(model.state_dict(), model_file_path)