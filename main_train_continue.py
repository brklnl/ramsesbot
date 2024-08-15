import torch
import os
import json
from torchvision import transforms
import wandb
from functools import partial
import signal

# Assuming these imports work the same as in your main.py
from dataset.dataset import Isaac_Data
from models.model import SilkiMaus
from utils.utils import plot_metrics, signal_handler, load_model
from loop.train_new import continue_training_from_checkpoint
from utils.utils_training.utils_losses.of_loss import one_scale
from utils.utils import create_directories_and_names, create_dataset_and_dataloader
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn as nn
from utils.utils_training.utils_transform.utils import A_transforms, torch_transforms

def load_checkpoint(model_path):
    return torch.load(model_path)

# Load configuration as in main.py
base_path = os.getcwd()
with open(os.path.join(base_path, 'master_thesis', 'main_test_2.1_isaac_data', 'configs', 'main_train_continue_config.json')) as f:
    config = json.load(f)

# Initialize Weights & Biases
run = wandb.init(project=config['project'], entity=config['entity'])

# Setup dataset and dataloaders as in main.py
train_a_transforms = A_transforms()
train_torch_transforms = transforms.Compose([transforms.ToTensor()])
val_torch_transforms = transforms.Compose([transforms.ToTensor()])
dataloader_train, dataloader_validation = create_dataset_and_dataloader(Isaac_Data, config, base_path, train_a_transforms, train_torch_transforms, val_torch_transforms)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model, optimizer, and scheduler as in main.py
model = SilkiMaus(num_classes=config['num_classes'])
optimizer = optim.Adam(model.parameters(), lr=config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'], betas=tuple(config['optimizer']['betas']))
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config['scheduler']['milestones'], gamma=config['scheduler']['gamma'])

# Load checkpoint here
checkpoint_path = config["checkpoint_path"]
checkpoint = load_checkpoint(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1  # Assuming you saved epoch in your checkpoint

# If you saved scheduler state, load it here
if 'scheduler_state_dict' in checkpoint:
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

# Continue with the rest of your setup as in main.py
segmentation_criterion = nn.CrossEntropyLoss()
flow_criterion = one_scale

full_model_metrics_dir, model_file_path, full_save_dir_training, \
full_save_dir_validation, full_plot_dir = create_directories_and_names(config,
                                                                       data_name=config['data_name'],
                                                                              date=config['date'],
                                                                                number=config['training_number'])
print(full_save_dir_training)
# Call the train_model function with start_epoch
# Ensure train_model is adapted to accept start_epoch
segmentation_accuracy_values, mae_of_values, segmentation_loss_values, average_flow_loss_values, total_loss_values, pixel_difference_values, iou_values, dice_coefficient_values = continue_training_from_checkpoint(
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

# Continue as in main.py
plot_metrics(segmentation_loss_values, average_flow_loss_values, total_loss_values, segmentation_accuracy_values, mae_of_values, full_plot_dir)
torch.save(model.state_dict(), model_file_path)