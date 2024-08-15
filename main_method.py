import torch
import torch.nn as nn
import json
import os
import signal

#######################################
from dataset.dataset import  TestConsecutiveImages 
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
from models.model import SilkiMaus
import os
import json
from utils.utils import create_video
from utils.utils_test.utils import test_model_and_lsd, save_output
from torchvision.transforms import InterpolationMode

base_path = os.getcwd()

# Load configuration with os.path.join for better path handling
with open(os.path.join(base_path,  'configs','main_method_config.json')) as f:
    config = json.load(f)

# Example usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if config['device'] == "auto" else torch.device(config['device'])
print(device)
model = SilkiMaus(num_classes=config['num_classes'])

# Load the checkpoint
checkpoint = torch.load(config['model_path'])

# Load the state_dict into the model
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()  # Set the model to evaluation mode

# Define the transformation with correct Resize usage and specify interpolation mode
transform = transforms.Compose([
    transforms.Resize((480, 640), interpolation=InterpolationMode.BILINEAR),  # Resize to 480x640 using BILINEAR interpolation
    transforms.ToTensor()
])

consecutive = TestConsecutiveImages(images_path = config['data_path'],
                                       transform=transform)

dataloader_validation = DataLoader(consecutive, batch_size=config['batch_size'], shuffle=config['shuffle'], num_workers=config['num_workers'])   
segmentation_criterion = nn.CrossEntropyLoss()
save_dir_validation = config['save_dir_validation']



# Ensure save directory exists
if not os.path.exists(save_dir_validation):
    os.makedirs(save_dir_validation)

def handle_interrupt(signal, frame):
    print("\nInterrupt received, creating video...")
    create_video(save_dir_validation, 'output_video.mp4', 5)
    exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, handle_interrupt)
try:
    # Call the function
    test_model_and_lsd(model, dataloader_validation, device, save_dir_validation, config)
finally:
    # Ensure create_video is called even if the script is interrupted
    create_video(save_dir_validation, 'output_video.mp4', 5)