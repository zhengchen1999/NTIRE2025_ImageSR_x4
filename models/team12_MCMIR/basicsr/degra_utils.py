from functools import partial
import torch
import numpy as np
from typing import List
from pathlib import Path
import os
import yaml
import ast
import torch
import copy
from ast import literal_eval
# from pykeops.torch import LazyTensor
import scipy
import scipy.linalg
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as v2
import argparse
from collections import defaultdict
import math
from random import randint
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from random import randint, seed
import warnings
warnings.filterwarnings("ignore", module="matplotlib\..*")


def gaussian_2d_kernel(sigma, size):
    """Generate a 2D Gaussian kernel."""
    x = torch.arange(-size // 2 + 1., size // 2 + 1.)
    y = torch.arange(-size // 2 + 1., size // 2 + 1.)
    x, y = torch.meshgrid(x, y)
    kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def upsample(x, sf):
    '''s-fold upsampler

    Upsampling the spatial size by filling the new entries with zeros

    x: tensor image, NxCxWxH
    '''
    st = 0
    z = torch.zeros(
        (x.shape[0],
         x.shape[1],
         x.shape[2] *
         sf,
         x.shape[3] *
         sf)).type_as(x)
    z[..., st::sf, st::sf].copy_(x)
    return z


def downsample(x, sf):
    '''s-fold downsampler

    Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others

    x: tensor image, NxCxWxH
    '''
    st = 0
    return x[..., st::sf, st::sf]


def gaussian_blur(x, sigma_blur, size_kernel):
    '''Blur a tensor image with Gaussian filter

    x: tensor image, NxCxWxH
    sigma: standard deviation of the Gaussian kernel
    '''
    kernel = gaussian_2d_kernel(sigma_blur, size_kernel).type_as(x)
    # uniform kernel
    kernel = kernel.view(1, 1, size_kernel, size_kernel)
    kernel = kernel.repeat(x.shape[1], 1, 1, 1)
    # kernel = kernel.flip(-1).flip(-2)
    return F.conv2d(x, kernel, stride=1, padding='same', groups=x.shape[1])


def square_mask(x, half_size_mask):
    """
    Black square mask of 20 x 20 pixels at the center of the image
    """
    d = x.shape[2] // 2

    mask = torch.ones_like(x)
    mask[:, :, d - half_size_mask:d + half_size_mask,
         d - half_size_mask:d + half_size_mask] = 0
    return mask * x


def paintbrush_mask(x):
    """
    Black mask that looks like paintbrush on the image. Make it random
    """
    mask_generator = MaskGenerator(x.shape[2], x.shape[3], 1, rand_seed=42)
    mask = torch.zeros_like(x)
    for i in range(x.shape[0]):
        mask_i = torch.from_numpy(
            mask_generator.sample().transpose((2, 0, 1))).to(x.device) - 1
        mask_i = (mask_i == 0).squeeze(0)
        mask[i] = mask_i
    return mask * x

class MaskGenerator():
    # copied from https://www.kaggle.com/code/tom99763/inpainting-mask-generator
    def __init__(self, height, width, channels=3, rand_seed=None, filepath=None):
        self.height = height
        self.width = width
        self.channels = channels
        self.filepath = filepath
        self.mask_files = []

        if self.filepath:
            filenames = [f for f in os.listdir(self.filepath)]
            self.mask_files = [f for f in filenames if any(
                filetype in f.lower() for filetype in ['.jpeg', '.png', '.jpg'])]
            print(">> Found {} masks in {}".format(
                len(self.mask_files), self.filepath))

        if rand_seed:
            seed(rand_seed)

    def _generate_mask(self):
        # print("height, width, channels", self.height, self.width, self.channels)
        img = np.zeros((self.height, self.width, self.channels), np.uint8)
        size = int((self.width + self.height) * 0.08)

        if self.width < 64 or self.height < 64:
            raise Exception("Width and Height of mask must be at least 64!")

        # Draw random lines
        for _ in range(10):
            x1, x2 = randint(self.width//2 - 30, self.width//2 +
                             30), randint(self.width//2 - 30, self.width//2 + 30)
            y1, y2 = randint(self.height//2 - 30, self.height//2 +
                             30), randint(self.height//2 - 30, self.height//2 + 30)
            thickness = randint(8, size)
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), thickness)
        return 1 - img

    def _load_mask(self, rotation=True, dilation=True, cropping=True):
        mask = cv2.imread(os.path.join(self.filepath, np.random.choice(
            self.mask_files, 1, replace=False)[0]))

        if rotation:
            rand = np.random.randint(-180, 180)
            M = cv2.getRotationMatrix2D(
                (mask.shape[1] / 2, mask.shape[0] / 2), rand, 1.5)
            mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))

        if dilation:
            rand = np.random.randint(5, 47)
            kernel = np.ones((rand, rand), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)

        if cropping:
            x = np.random.randint(0, mask.shape[1] - self.width)
            y = np.random.randint(0, mask.shape[0] - self.height)
            mask = mask[y:y + self.height, x:x + self.width]

        return (mask > 1).astype(np.uint8)

    def sample(self, random_seed=None):
        if random_seed:
            seed(random_seed)
        if self.filepath and len(self.mask_files) > 0:
            return self._load_mask()
        else:
            return self._generate_mask()


def random_mask(x, p, seed=None):
    """
    Random mask on x
    """
    np.random.seed(42)
    mask = torch.from_numpy(np.random.binomial(n=1, p=1-p, size=(
        x.shape[0], x.shape[2], x.shape[3]))).to(x.device)

    return mask.unsqueeze(1) * x


# comes from deepinv
def bicubic_filter(factor=2):
    r"""
    Bicubic filter.

    It has size (4*factor, 4*factor) and is defined as

    .. math::

        \begin{equation*}
            w(x, y) = \begin{cases}
                (a + 2)|x|^3 - (a + 3)|x|^2 + 1 & \text{if } |x| \leq 1 \\
                a|x|^3 - 5a|x|^2 + 8a|x| - 4a & \text{if } 1 < |x| < 2 \\
                0 & \text{otherwise}
            \end{cases}
        \end{equation*}

    for :math:`x, y \in {-2\text{factor} + 0.5, -2\text{factor} + 0.5 + 1/\text{factor}, \ldots, 2\text{factor} - 0.5}`.

    :param int factor: downsampling factor
    """
    x = np.arange(start=-2 * factor + 0.5, stop=2 * factor, step=1) / factor
    a = -0.5
    x = np.abs(x)
    w = ((a + 2) * np.power(x, 3) - (a + 3) * np.power(x, 2) + 1) * (x <= 1)
    w += (
        (a * np.power(x, 3) - 5 * a * np.power(x, 2) + 8 * a * x - 4 * a)
        * (x > 1)
        * (x < 2)
    )
    w = np.outer(w, w)
    w = w / np.sum(w)
    return torch.Tensor(w).unsqueeze(0).unsqueeze(0)

# Function to create the downsampling matrix
def create_downsampling_matrix(H, W, sf, device):
    assert H % sf == 0 and W % sf == 0, "Image dimensions must be divisible by sf"

    H_ds, W_ds = H // sf, W // sf  # Downsampled dimensions
    N = H * W  # Total number of pixels in the original image
    M = H_ds * W_ds  # Total number of pixels in the downsampled image

    # Initialize downsampling matrix of size (M, N)
    downsample_matrix = torch.zeros((M, N), device=device)

    # Fill the matrix with 1s at positions corresponding to downsampling
    for i in range(H_ds):
        for j in range(W_ds):
            # The index in the downsampled matrix
            downsampled_idx = i * W_ds + j

            # The corresponding index in the original flattened matrix
            original_idx = (i * sf * W) + (j * sf)

            # Set the value to 1 to perform downsampling
            downsample_matrix[downsampled_idx, original_idx] = 1

    return downsample_matrix