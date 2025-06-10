from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
from pathlib import Path
from PIL import Image
import os
import torchvision.transforms.functional as TF
from itertools import permutations
import numpy as np
import cv2
import random   
import json
from torch.utils.data.dataset import Dataset
import torch
import math
import torch.nn.functional as F
from .degradation import *
from pathlib import Path

class TestSetLoader(Dataset):
    def __init__(self, input_path):
        super(TestSetLoader, self).__init__()
        self.dataset_dir = input_path

        # TODO file paths
        self.file_list = sorted(self.get_all_image_paths(self.dataset_dir))
        
    def get_all_image_paths(self, root_dir, extensions=("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff")):
        """
        Recursively finds all image file paths under the root directory, including those in subdirectories.
        """
        return sorted([str(file) for ext in extensions for file in Path(root_dir).rglob(ext)])

    def __getitem__(self, index):
        lr_path = self.file_list[index]
        rgb_lr = cv2.imread(lr_path)
        lr_rgb_img = np.array(cv2.cvtColor(rgb_lr, cv2.COLOR_BGR2RGB)).astype(np.float32)
        low = torch.from_numpy(lr_rgb_img.copy()).permute(2, 0, 1).unsqueeze(0)
        low = low / 255     
        img_lr =  low
        img_lr =  img_lr.squeeze(0)
        return img_lr.contiguous(), lr_path 

    def __len__(self):
        return len(self.file_list)