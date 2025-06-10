import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from torchvision import transforms
import torchvision.transforms.functional as TF

from basicsr.data.transforms import augment, paired_random_crop
# from starter_kit.degradations import simple_deg_simulation
import utils.augmentation as augmentation
import math
# from starter_kit.imutils import downsample_raw, convert_to_tensor
import json
import timm  # Ensure timm is installed
from timm.data.loader import create_loader, PrefetchLoader
from timm.data.loader import *
import cv2
import torch.nn.functional as F


class LSDIRLoader(Dataset):
    def __init__(self, cfg):
        super(LSDIRLoader, self).__init__()

        self.data_root = cfg.data_dir  # Base directory
        self.outputSize = cfg.img_size
        self.transform = transforms.Compose([transforms.ToTensor()])

        # Load file paths from JSON
        with open(cfg.json_path, 'r') as f:
            self.file_list = json.load(f)

    def __getitem__(self, index):
        # Get file paths from JSON
        hr_path = os.path.join(self.data_root, self.file_list[index]["path_gt"])
        lr_path = os.path.join(self.data_root, self.file_list[index]["path_lq"])

        # Read images
        rgb_hr = cv2.imread(hr_path)
        rgb_lr = cv2.imread(lr_path)
        
        # Convert BGR to RGB
        hr_rgb_img = cv2.cvtColor(rgb_hr, cv2.COLOR_BGR2RGB).astype(np.float32)
        lr_rgb_img = cv2.cvtColor(rgb_lr, cv2.COLOR_BGR2RGB).astype(np.float32)

        # Convert to torch tensors
        hr_rgb_final = torch.from_numpy(hr_rgb_img.copy()).permute(2, 0, 1).unsqueeze(0)
        lr_rgb_final = torch.from_numpy(lr_rgb_img.copy()).permute(2, 0, 1).unsqueeze(0)

        _, _, H, W = lr_rgb_final.shape
        lr_rgb_final = F.interpolate(lr_rgb_final, (max(self.outputSize, H), max(self.outputSize, W)), mode='bilinear', align_corners=False)
        hr_rgb_final = F.interpolate(hr_rgb_final, (max(self.outputSize, 4 * H), max(self.outputSize, 4 * W)), mode='bilinear', align_corners=False)

        # Normalize images to [0, 1]
        lr_rgb_final = lr_rgb_final / 255.0
        hr_rgb_final = hr_rgb_final / 255.0

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(lr_rgb_final, output_size=(self.outputSize, self.outputSize))
        low = TF.crop(lr_rgb_final, i, j, h, w)
        high = TF.crop(hr_rgb_final, i * 4, j * 4, 4 * h, 4 * w)

        # Apply augmentations
        img_hr, img_lr = augmentation.apply_augment(
            high, low, ["rgb", "mixup", "vertical", "horizontal", "none"],
            [1.0, 1.0, 1.0, 1.0, 1.0], [0.35, 0.35, 0.1, 0.1, 0.1], 1.2, 1.2, None
        )

        img_hr, img_lr = img_hr.squeeze(0), img_lr.squeeze(0)

        return img_lr, img_hr

    def __len__(self):
        return len(self.file_list) 


class ValLSDIRLoader(Dataset):
    def __init__(self, cfg_test):
        super(ValLSDIRLoader, self).__init__()
        self.hr_dataset_dir = os.path.join(cfg_test.data_dir, 'HR', 'val')
        self.lr_dataset_dir = os.path.join(cfg_test.data_dir, 'X4', 'val')
        self.hr_file_list = sorted(os.listdir(self.hr_dataset_dir))
        self.lr_file_list = sorted(os.listdir(self.lr_dataset_dir))
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.outputSize = cfg_test.img_size
    def __getitem__(self, index):
        hr_path = os.path.join(self.hr_dataset_dir, self.hr_file_list[index])
        lr_path = os.path.join(self.lr_dataset_dir, self.lr_file_list[index])
        rgb_hr = cv2.imread(hr_path)
        rgb_lr = cv2.imread(lr_path)
        
        hr_rgb_img = np.array(cv2.cvtColor(rgb_hr, cv2.COLOR_BGR2RGB)) 
        lr_rgb_img = np.array(cv2.cvtColor(rgb_lr, cv2.COLOR_BGR2RGB))
        # TODO check valid 255 rgb range
        hr_rgb_img = (hr_rgb_img).astype(np.float32)
        lr_rgb_img = (lr_rgb_img).astype(np.float32)  
        hr_rgb_final = torch.from_numpy(hr_rgb_img.copy()).permute(2, 0, 1).unsqueeze(0)
        lr_rgb_final = torch.from_numpy(lr_rgb_img.copy()).permute(2, 0, 1).unsqueeze(0)
        _,_,H,W = lr_rgb_final.shape
        lr_rgb_final = F.interpolate(lr_rgb_final,(max(self.outputSize,H),max(self.outputSize,W)),mode='bilinear',align_corners=False)
        hr_rgb_final = F.interpolate(hr_rgb_final,(max(self.outputSize,4*H),max(self.outputSize,4*W)),mode='bilinear',align_corners=False)
        
        lr_rgb_final = lr_rgb_final / 255
        hr_rgb_final = hr_rgb_final / 255
        
        i,j,h,w = transforms.RandomCrop.get_params(lr_rgb_final, output_size = (self.outputSize,self.outputSize))
        low = TF.crop(lr_rgb_final, i, j, h, w)
        high = TF.crop(hr_rgb_final, i*4, j*4, 4*h, 4*w)
        img_hr, img_lr = augmentation.apply_augment(high, low, ["rgb", "mixup", "vertical", "horizontal", "none"], [1.0, 1.0, 1.0, 1.0, 1.0], [0.35, 0.35, 0.1, 0.1, 0.1],
                1.2, 1.2, None)
        img_hr, img_lr = img_hr.squeeze(0), img_lr.squeeze(0)
        return img_lr, img_hr
    
    def __len__(self):
        return min(len(self.hr_file_list),len(self.lr_file_list))
    
class TrainSetLoader(Dataset):
    def __init__(self, cfg):
        super(TrainSetLoader, self).__init__()

        self.hr_dataset_dir = cfg.data_dir +'/train/HR/DIV2K_train_HR/'
        self.lr_dataset_dir = cfg.data_dir +'/train/LR/DIV2K_train_LR_mild/'
        # self.kernel_path = np.load(cfg.kernel_path, allow_pickle=True)
        self.hr_file_list = sorted(os.listdir(self.hr_dataset_dir))
        self.lr_file_list = sorted(os.listdir(self.lr_dataset_dir))
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.outputSize = cfg.img_size
        
    def __getitem__(self, index):
        hr_path = self.hr_dataset_dir + self.hr_file_list[index]
        lr_path = self.lr_dataset_dir + self.lr_file_list[index]
        # raw_lr, raw_hr = np.load(lr_path), np.load(hr_path)
        rgb_hr = cv2.imread(hr_path)
        rgb_lr = cv2.imread(lr_path)
        
        hr_rgb_img = np.array(cv2.cvtColor(rgb_hr, cv2.COLOR_BGR2RGB)) 
        lr_rgb_img = np.array(cv2.cvtColor(rgb_lr, cv2.COLOR_BGR2RGB))
        # TODO check valid 255 rgb range
        hr_rgb_img = (hr_rgb_img ).astype(np.float32)
        lr_rgb_img = (lr_rgb_img ).astype(np.float32)

        hr_rgb_final = torch.from_numpy(hr_rgb_img.copy()).permute(2, 0, 1).unsqueeze(0)
        lr_rgb_final = torch.from_numpy(lr_rgb_img.copy()).permute(2, 0, 1).unsqueeze(0)
        _,_,H,W = lr_rgb_final.shape
        
        # h = 100 eh = 128, h -> eh, ratio = eh/h  w = 300, w -> 300*ratio
        # lr_rgb_final = F.interpolate(lr_rgb_final,(max(self.outputSize,H),max(self.outputSize,W)),mode='bilinear',align_corners=False)
        # hr_rgb_final = F.interpolate(hr_rgb_final,(max(self.outputSize,4*H),max(self.outputSize,4*W)),mode='bilinear',align_corners=False)
        
        spatial_arr = [H,W]
        if min(H,W) < self.outputSize:
            min_index = spatial_arr.index(min(spatial_arr))
            ratio = self.outputSize / spatial_arr[min_index]
            lr_rgb_final = F.interpolate(lr_rgb_final,(math.ceil(H*ratio),math.ceil(W*ratio)),mode='bilinear',align_corners=False)
            hr_rgb_final = F.interpolate(hr_rgb_final,(math.ceil(4*H*ratio),math.ceil(4*W*ratio)),mode='bilinear',align_corners=False)
        
        
        lr_rgb_final = lr_rgb_final / 255
        hr_rgb_final = hr_rgb_final / 255
        i,j,h,w = transforms.RandomCrop.get_params(lr_rgb_final, output_size = (self.outputSize,self.outputSize))
        low = TF.crop(lr_rgb_final, i, j, h, w)
        high = TF.crop(hr_rgb_final, i*4, j*4, 4*h, 4*w)
        img_hr, img_lr = augmentation.apply_augment(high, low, ["rgb", "mixup", "vertical", "horizontal", "none"], [1.0, 1.0, 1.0, 1.0, 1.0], [0.35, 0.35, 0.1, 0.1, 0.1],
                1.2, 1.2, None)
        img_hr, img_lr = img_hr.squeeze(0), img_lr.squeeze(0)
        return  img_lr, img_hr

    def __len__(self):
        return min(len(self.hr_file_list),len(self.lr_file_list))

class ValidSetLoader(Dataset):
    def __init__(self, cfg_test):
        super(ValidSetLoader, self).__init__()
        self.hr_dataset_dir = cfg_test.data_dir +'/val/HR/'
        self.lr_dataset_dir = cfg_test.data_dir +'/val/LR/DIV2K_valid_LR_mild/'
        self.hr_file_list = sorted(os.listdir(self.hr_dataset_dir))
        self.lr_file_list = sorted(os.listdir(self.lr_dataset_dir))
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.outputSize = cfg_test.img_size
    def __getitem__(self, index):
        hr_path = self.hr_dataset_dir + self.hr_file_list[index]
        lr_path = self.lr_dataset_dir + self.lr_file_list[index]
        rgb_hr = cv2.imread(hr_path)
        rgb_lr = cv2.imread(lr_path)
        
        hr_rgb_img = np.array(cv2.cvtColor(rgb_hr, cv2.COLOR_BGR2RGB)) 
        lr_rgb_img = np.array(cv2.cvtColor(rgb_lr, cv2.COLOR_BGR2RGB))
        # TODO check valid 255 rgb range
        hr_rgb_img = (hr_rgb_img).astype(np.float32)
        lr_rgb_img = (lr_rgb_img).astype(np.float32)  
        hr_rgb_final = torch.from_numpy(hr_rgb_img.copy()).permute(2, 0, 1).unsqueeze(0)
        lr_rgb_final = torch.from_numpy(lr_rgb_img.copy()).permute(2, 0, 1).unsqueeze(0)
        _,_,H,W = lr_rgb_final.shape
        lr_rgb_final = F.interpolate(lr_rgb_final,(max(self.outputSize,H),max(self.outputSize,W)),mode='bilinear',align_corners=False)
        hr_rgb_final = F.interpolate(hr_rgb_final,(max(self.outputSize,4*H),max(self.outputSize,4*W)),mode='bilinear',align_corners=False)
        
        lr_rgb_final = lr_rgb_final / 255
        hr_rgb_final = hr_rgb_final / 255
        
        i,j,h,w = transforms.RandomCrop.get_params(lr_rgb_final, output_size = (self.outputSize,self.outputSize))
        low = TF.crop(lr_rgb_final, i, j, h, w)
        high = TF.crop(hr_rgb_final, i*4, j*4, 4*h, 4*w)
        img_hr, img_lr = augmentation.apply_augment(high, low, ["rgb", "mixup", "vertical", "horizontal", "none"], [1.0, 1.0, 1.0, 1.0, 1.0], [0.35, 0.35, 0.1, 0.1, 0.1],
                1.2, 1.2, None)
        img_hr, img_lr = img_hr.squeeze(0), img_lr.squeeze(0)
        return img_lr, img_hr
    
    def __len__(self):
        return min(len(self.hr_file_list),len(self.lr_file_list))

def create_rggb_loader(
        dataset,
        input_size,
        batch_size,
        is_training=False,
        use_prefetcher= False,
        no_aug=False,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_split=False,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        color_jitter=None,
        auto_augment=None,
        num_aug_repeats=0,
        num_aug_splits=0,
        interpolation='bilinear',
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        num_workers=4,
        distributed=False,
        collate_fn=None,
        pin_memory=True,
        use_multi_epochs_loader=False,
        worker_seeding='all',
        crop_pct=None,
        img_dtype=torch.float32,
        device=torch.device('cuda'),
        re_num_splits=0
    ):

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=4,  # Adjust based on your system's capabilities
        pin_memory=pin_memory,
        drop_last=is_training
    )


    if use_prefetcher:
        # Apply prefetching with all specified configurations
        prefetch_re_prob = re_prob if is_training else 0.
        loader = PrefetchLoader(
            loader,
            mean=mean,
            std=std,
            channels=4,  # Ensure this matches the number of channels in your images
            device=device,
            img_dtype=img_dtype,
            re_prob=prefetch_re_prob,
            re_mode=re_mode,
            re_count=re_count,
            re_num_splits=re_num_splits  # Control random erase application across batch parts if using augmentation splits
        )

    return loader

def pad_to_min_size(image, min_size=224, padding_mode="constant", padding_value=0.0):
    """
    Pads an image tensor (B, C, H, W) to ensure minimum size in both height and width.

    Args:
        image (torch.Tensor): Input tensor of shape (B, C, H, W).
        min_size (int): The minimum required height and width (default: 224).
        padding_mode (str): Padding mode ('constant', 'reflect', 'replicate', 'circular').
        padding_value (float): Padding value (only for 'constant' mode).
    
    Returns:
        torch.Tensor: Padded image tensor with (B, C, new_H, new_W).
    """
    B, C, H, W = image.shape

    # Compute required padding
    pad_h = max(0, min_size - H)  # Only add padding if H < min_size
    pad_w = max(0, min_size - W)  # Only add padding if W < min_size

    # Padding is applied as (left, right, top, bottom)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # Apply padding
    padded_image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode=padding_mode, value=padding_value)

    return padded_image
