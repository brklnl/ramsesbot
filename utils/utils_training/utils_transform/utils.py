import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
from torchvision.transforms import ToTensor
import torch

def A_transforms():
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0),
        A.GaussianBlur(sigma_limit=9, p=0.5),
        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), mean=0, always_apply=False, p=0.5),
        #A.Blur(blur_limit=3, always_apply=False, p=0.5),
        #A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #ToTensorV2()
          ])

def torch_transforms():
    return ToTensor()

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    return tensor * std + mean