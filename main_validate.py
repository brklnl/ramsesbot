import torch
import torch.nn as nn
import json
import os
#######################################
from dataset.dataset import Isaac_Data 
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
from models.model import SilkiMaus
import os

from utils.utils import create_video
from utils.utils_validate.utils import validate_model

with open('/home/ist-sim/burak/SEGFLOW_master_thesis/master_thesis/main_test_2.1_isaac_data/configs/main_validate_config.json') as f:
    config = json.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SilkiMaus(config["num_classes"]).to(device)
# Load the checkpoint
# Load the checkpoint
checkpoint = torch.load(config['model_path'])

# Load the weights into the model
model.load_state_dict(checkpoint['model_state_dict'])

# Load the epoch number
epoch = checkpoint['epoch']




model.eval()  # Set the model to evaluation mode


val_torch_transforms = transforms.Compose([transforms.ToTensor()])


ISAAC_dataset_validation = Isaac_Data(
    mask_flow_path=config["dataset"]["mask_flow_path"],
    segmentation_images_path=config["dataset"]["segmentation_images_path"],
    segmentation_mask_path=config["dataset"]["segmentation_mask_path"],
    a_transforms=None,
    torch_transforms=val_torch_transforms,
    start_index=config["dataset"]["start_index"]
)

dataloader_validation = DataLoader(ISAAC_dataset_validation, batch_size=config["batch_size"], shuffle=False)
segmentation_criterion = nn.CrossEntropyLoss()

save_dir_validation = config["save_dir_validation"]
if not os.path.exists(save_dir_validation):
    os.makedirs(save_dir_validation)
    



# Call the function
validate_model(model, dataloader_validation, device, segmentation_criterion, save_dir_validation,config)

create_video(save_dir_validation, os.path.join(save_dir_validation, 'combined.mp4'),20)