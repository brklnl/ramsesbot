import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from utils.utils_training.utils_transform.utils import A_transforms, torch_transforms
from albumentations.core.composition import ReplayCompose
import logging  
import re
import cv2

class TartanAir_1(Dataset):
    def __init__(self, mask_flow_path, segmentation_images_path, segmentation_mask_path, a_transforms=A_transforms(), torch_transforms=torch_transforms()):
        self.mask_flow_path = mask_flow_path
        self.segmentation_images_path = segmentation_images_path
        self.segmentation_mask_path = segmentation_mask_path
        self.a_transforms = a_transforms
        self.torch_transforms = torch_transforms

        self.mask_files = sorted(os.listdir(mask_flow_path))
        self.seg_files = sorted(os.listdir(segmentation_images_path))

    def __len__(self):
        return len(self.mask_files)

    def __getitem__(self, idx):
        # Set the index
        idx += self.index_offset

        # Load flow mask
        mask_filename = f"{idx:06d}_{idx+1:06d}_flow.npy"
        flow_mask = np.load(os.path.join(self.mask_flow_path, mask_filename))

        # Load segmentation image and mask
        segmentation_image_filename = f"{idx:06d}_left.png"
        segmentation_image_path = os.path.join(self.segmentation_images_path, segmentation_image_filename)
        segmentation_image = Image.open(segmentation_image_path).convert('RGB')

        # Load segmentation mask
        segmentation_mask_filename = f"{idx:06d}_left_seg.npy"
        segmentation_mask_path = os.path.join(self.segmentation_mask_path, segmentation_mask_filename)
        segmentation_mask = np.load(os.path.join(self.segmentation_mask_path, segmentation_mask_path))

        # Convert segmentation mask and optical flow mask to tensor
        segmentation_mask = torch.from_numpy(np.array(segmentation_mask)).int()
        flow_mask = torch.from_numpy(flow_mask).permute(2,0,1)


        # stick with the original images
        #next_idx = (idx + 1)
        total_images = len(self.seg_files)

        # Apply the offset to the index for current and next images
        current_idx = idx 
        next_idx = (idx + 1 )

        current_flow_filename = f"{current_idx:06d}_left.png"

        current_image_path = os.path.join(self.segmentation_images_path, current_flow_filename)
        current_image = Image.open(current_image_path).convert('RGB')

        # Check if next_idx is within range
        if next_idx < total_images:
            next_flow_filename = f"{next_idx:06d}_left.png"
            next_image_path = os.path.join(self.segmentation_images_path, next_flow_filename)
            next_image = Image.open(next_image_path).convert('RGB')
        else:

            next_image = current_image

        current_image_array = np.array(current_image)  # Shape: [H, W, 3]
        next_image_array = np.array(next_image)  # Shape: [H, W, 3]

        print("\n##################   Checking Transformation   ################################\n")
        # Apply transformations if specified
        # if self.a_transforms is not None:
        #     # Ensure self.a_transforms is a list
        #     if not isinstance(self.a_transforms, list):
        #         self.a_transforms = [self.a_transforms]

        #     # Convert Albumentations transformations to ReplayCompose
        #     replay_transforms = ReplayCompose(self.a_transforms)

        #     # Apply the same random transformation to all three images
        #     transformed = replay_transforms(image=current_image_array)
        #     current_image_array = transformed["image"]
        #     replay = transformed["replay"]

        #     next_image_array = replay_transforms.replay(replay, image=next_image_array)["image"]
        #     segmentation_image = replay_transforms.replay(replay, image=np.array(segmentation_image))["image"]
        if self.a_transforms is not None:
            # Apply the random transformation only to the segmentation image
            transformed = self.a_transforms(image=np.array(segmentation_image))
            segmentation_image = transformed["image"]

        if self.torch_transforms is not None:
            # Convert the images to PyTorch tensors using Torchvision's ToTensor
            current_image_array = self.torch_transforms(current_image_array)
            next_image_array = self.torch_transforms(next_image_array)
            segmentation_image = self.torch_transforms(segmentation_image)

            stacked_flow = torch.cat([current_image_array, next_image_array], dim=0)
        else:
            print("Transform is not applied.")


        
        print("\n###########   Dataset Class is Done   #########\n")
        print("###########   DataLoader is Working   #############")

        return {
            'stacked_flow': stacked_flow.float(),
            'flow_mask': flow_mask,
            'segmentation_image': segmentation_image.float(),
            'segmentation_mask': segmentation_mask
        }
    
'''
############# ISAAC DATA ####################
After loading the images the sizes:
Flow mask shape: (480, 640, 4)
Segmentation image size: (640, 480)
Segmentation mask shape: (480, 640)

Stacked flow image is produced after the transformations thats why its size:
Stacked flow shape: torch.Size([6, 480, 640])

After transformation is applied the images sizes:
Shape of Stacked Flow:  torch.Size([6, 480, 640])
Shape of Segmentation Image:  torch.Size([3, 480, 640])
Shape of Flow Mask:  torch.Size([4, 480, 640])
Shape of Segmentation Mask:  torch.Size([480, 640])
'''
class Isaac_Data(Dataset):
    def __init__(self, mask_flow_path, segmentation_images_path, segmentation_mask_path, a_transforms=A_transforms(), torch_transforms=torch_transforms(), start_index=0):
        self.mask_flow_path = mask_flow_path
        self.segmentation_images_path = segmentation_images_path
        self.segmentation_mask_path = segmentation_mask_path
        self.a_transforms = a_transforms
        self.torch_transforms = torch_transforms
        self.index_offset = start_index  # Add this line


        self.mask_files = sorted(os.listdir(mask_flow_path))
        self.seg_files = sorted(os.listdir(segmentation_images_path))

    def __len__(self):
        return len(self.mask_files)

    def __getitem__(self, idx):
        # Set the index
        idx += self.index_offset

        # Load flow mask
        mask_filename = f"motion_vectors_{idx:06d}.npy"
        flow_mask = np.load(os.path.join(self.mask_flow_path, mask_filename))

        # Load segmentation image
        segmentation_image_filename = f"rgb_{idx:06d}.png"
        segmentation_image_path = os.path.join(self.segmentation_images_path, segmentation_image_filename)
        segmentation_image = Image.open(segmentation_image_path).convert('RGB')

        # Load segmentation mask
        segmentation_mask_filename = f"semantic_segmentation_{idx:06d}.npy"
        segmentation_mask_path = os.path.join(self.segmentation_mask_path, segmentation_mask_filename)
        segmentation_mask = np.load(os.path.join(self.segmentation_mask_path, segmentation_mask_path))
        segmentation_mask = np.where(segmentation_mask == 7, 0, segmentation_mask)


        # Convert segmentation mask and optical flow mask to tensor
        segmentation_mask = torch.from_numpy(np.array(segmentation_mask)).int()
        flow_mask = torch.from_numpy(flow_mask).permute(2,0,1)
        flow_mask = flow_mask[:2, :, :]


        # stick with the original images
        total_images = len(self.seg_files)

        # Apply the offset to the index for current and next images
        current_idx = idx 
        next_idx = (idx + 1 )

        current_flow_filename = f"rgb_{current_idx:06d}.png"

        current_image_path = os.path.join(self.segmentation_images_path, current_flow_filename)
        current_image = Image.open(current_image_path).convert('RGB')

        # Check if next_idx is within range
        if next_idx < total_images:
            next_flow_filename = f"rgb_{next_idx:06d}.png"
            next_image_path = os.path.join(self.segmentation_images_path, next_flow_filename)
            next_image = Image.open(next_image_path).convert('RGB')
        else:

            next_image = current_image

        current_image_array = np.array(current_image)  # Shape: [H, W, 3]
        next_image_array = np.array(next_image)  # Shape: [H, W, 3]

        #print("\n##################   Checking Transformation   ################################\n")

        if self.a_transforms is not None:
            # Apply the random transformation only to the segmentation image
            transformed = self.a_transforms(image=np.array(segmentation_image))
            segmentation_image = transformed["image"]

        if self.torch_transforms is not None:
            # Convert the images to PyTorch tensors using Torchvision's ToTensor
            current_image_array = self.torch_transforms(current_image_array)
            next_image_array = self.torch_transforms(next_image_array)
            segmentation_image = self.torch_transforms(segmentation_image)

            stacked_flow = torch.cat([current_image_array, next_image_array], dim=0)
        else:
            print("Transform is not applied.")


        
        #print("\n###########   Dataset Class is Done   #########\n")
        #print("###########   DataLoader is Working   #############")

        return {
            'stacked_flow': stacked_flow.float(),
            'flow_mask': flow_mask,
            'segmentation_image': segmentation_image.float(),
            'segmentation_mask': segmentation_mask.long()
        }


class Isaac_Data_2(Dataset):
    def __init__(self, mask_flow_path, segmentation_images_path, segmentation_mask_path, a_transforms=A_transforms(), torch_transforms=torch_transforms(), start_index=0, image_type='RGB', image_size=(256, 256), flow_mask_filename_format="motion_vectors_{:04d}.npy", image_filename_format="rgb_{:04d}.png", segmentation_mask_filename_format="semantic_segmentation_{:04d}.npy"):
        self.mask_flow_path = mask_flow_path
        self.segmentation_images_path = segmentation_images_path
        self.segmentation_mask_path = segmentation_mask_path
        self.a_transforms = a_transforms
        self.torch_transforms = torch_transforms
        self.index_offset = start_index
        self.image_type = image_type
        self.image_size = image_size
        self.flow_mask_filename_format = flow_mask_filename_format
        self.image_filename_format = image_filename_format
        self.segmentation_mask_filename_format = segmentation_mask_filename_format

        self.mask_files = sorted(os.listdir(mask_flow_path))
        self.seg_files = sorted(os.listdir(segmentation_images_path))

    def __len__(self):
        return len(self.mask_files)

    def __getitem__(self, idx):
        idx += self.index_offset

        mask_filename = self.flow_mask_filename_format.format(idx)
        flow_mask = self.load_mask(self.mask_flow_path, mask_filename)

        segmentation_image_filename = self.image_filename_format.format(idx)
        segmentation_image = self.load_image(self.segmentation_images_path, segmentation_image_filename)

        segmentation_mask_filename = self.segmentation_mask_filename_format.format(idx)
        segmentation_mask = self.load_mask(self.segmentation_mask_path, segmentation_mask_filename)

        segmentation_mask = torch.from_numpy(np.array(segmentation_mask)).int()
        flow_mask = torch.from_numpy(flow_mask).permute(2,0,1)
        flow_mask = flow_mask[:2, :, :]

        total_images = len(self.seg_files)

        current_idx = idx 
        next_idx = (idx + 1 )

        current_flow_filename = self.image_filename_format.format(current_idx)
        current_image = self.load_image(self.segmentation_images_path, current_flow_filename)

        if next_idx < total_images:
            next_flow_filename = self.image_filename_format.format(next_idx)
            next_image = self.load_image(self.segmentation_images_path, next_flow_filename)
        else:
            next_image = current_image

        current_image_array = np.array(current_image)
        next_image_array = np.array(next_image)

        if self.a_transforms is not None:
            transformed = self.a_transforms(image=np.array(segmentation_image))
            segmentation_image = transformed["image"]

        if self.torch_transforms is not None:
            current_image_array = self.torch_transforms(current_image_array)
            next_image_array = self.torch_transforms(next_image_array)
            segmentation_image = self.torch_transforms(segmentation_image)

            stacked_flow = torch.cat([current_image_array, next_image_array], dim=0)

        return {
            'stacked_flow': stacked_flow.float(),
            'flow_mask': flow_mask,
            'segmentation_image': segmentation_image.float(),
            'segmentation_mask': segmentation_mask
        }

    def load_image(self, path, filename):
        try:
            image_path = os.path.join(path, filename)
            return Image.open(image_path).convert(self.image_type).resize(self.image_size)
        except Exception as e:
            logging.error(f"Failed to load image from {image_path}: {e}")
            raise

    def load_mask(self, path, filename):
        try:
            mask_path = os.path.join(path, filename)
            return np.load(mask_path)
        except Exception as e:
            logging.error(f"Failed to load mask from {mask_path}: {e}")
            raise



import os
import re
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import tempfile

import os
import re
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import tempfile

class TestConsecutiveImages(Dataset):
    def __init__(self, images_path, transform=None):
        self.images_path = images_path
        self.transform = transform
        self.image_files = []

        if os.path.isdir(images_path):
            # Load images from directory
            self.image_files = sorted(os.listdir(images_path), key=lambda x: int(re.search(r'\d+', x).group()))
            self.image_files = [os.path.join(images_path, f) for f in self.image_files]  # Prepend directory path
        elif os.path.isfile(images_path) and images_path.endswith(('.mp4', '.avi', '.mov')):
            # Create a temporary directory to store frames
            self.temp_dir = tempfile.TemporaryDirectory()
            cap = cv2.VideoCapture(images_path)
            success, frame = cap.read()
            frame_idx = 0
            while success:
                frame_path = os.path.join(self.temp_dir.name, f'frame_{frame_idx:06d}.jpg')
                cv2.imwrite(frame_path, frame)
                self.image_files.append(frame_path)
                success, frame = cap.read()
                frame_idx += 1
            cap.release()
        else:
            raise ValueError("Invalid images_path. It should be a directory of images or a video file.")

    def __len__(self):
        # Subtract 1 because we can't get a consecutive image for the last image
        return len(self.image_files) - 1

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        next_image_file = self.image_files[idx + 1]
        # Debug: Print the file paths being accessed
        # print(f"Accessing image: {image_file}")
        # print(f"Accessing next image: {next_image_file}")
        image = Image.open(image_file).convert('RGB')
        next_image = Image.open(next_image_file).convert('RGB')

        if self.transform:
            image = self.transform(image)
            next_image = self.transform(next_image)

        # Stack the current image and the next image along the channel dimension
        stacked_images = torch.cat((image, next_image), dim=0)

        return {'image': image, 'stacked_images': stacked_images}

    def __del__(self):
        # Clean up the temporary directory
        if hasattr(self, 'temp_dir'):
            self.temp_dir.cleanup()


# class TestConsecutiveImages(Dataset):
#     def __init__(self, images_path, transform=None):
#         self.images_path = images_path
#         self.transform = transform
#         self.image_files = sorted(os.listdir(images_path), key=lambda x: int(re.search(r'\d+', x).group()))

#     def __len__(self):
#         # Subtract 1 because we can't get a consecutive image for the last image
#         return len(self.image_files) - 1

#     def __getitem__(self, idx):
#         image_file = self.image_files[idx]
#         next_image_file = self.image_files[idx + 1]

#         image = Image.open(os.path.join(self.images_path, image_file)).convert('RGB')
#         next_image = Image.open(os.path.join(self.images_path, next_image_file)).convert('RGB')

#         if self.transform:
#             image = self.transform(image)
#             next_image = self.transform(next_image)

#         # Stack the current image and the next image along the channel dimension
#         stacked_images = torch.cat((image, next_image), dim=0)

#         return {'image': image, 'stacked_images': stacked_images}
