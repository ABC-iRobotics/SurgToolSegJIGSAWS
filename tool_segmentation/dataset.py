
"""

Implementation from https://github.com/ternaus/robot-surgery-segmentation

"""

import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import img_to_tensor

binary_factor = 255

class RoboticsDatasetTrain(Dataset):
    def __init__(self, file_names, to_augment=False, transform=None):
        self.file_names = file_names
        self.to_augment = to_augment
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_name = self.file_names[idx]
        image = load_image(img_file_name)
        mask = load_mask(img_file_name)

        data = {"image": image, "mask": mask}
        augmented = self.transform(**data)
        image, mask = augmented["image"], augmented["mask"]


        return img_to_tensor(image), torch.from_numpy(np.expand_dims(mask, 0)).float() 


class RoboticsDatasetPred(Dataset):
    def __init__(self, file_names, transform=None):
        self.file_names = file_names
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_name = self.file_names[idx]
        image = load_image(img_file_name)

        data = {"image": image}
        augmented = self.transform(**data)
        image = augmented["image"]

        return img_to_tensor(image), str(img_file_name)


def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path):

    mask_folder = 'binary_masks' 
    factor = binary_factor 

    mask = cv2.imread(str(path).replace('images', mask_folder).replace('jpg', 'png'), 0)

    return (mask / factor).astype(np.uint8)

