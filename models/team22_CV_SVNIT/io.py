import os.path
import logging
import torch
import argparse
import json
import glob

from pprint import pprint
from utils.model_summary import get_model_flops
from utils import utils_logger
from utils import utils_image as util

from models.team22_HAT.HAT import HAT

def forward(img_lq, model, tile=None, tile_overlap=32, scale=4):
    if tile is None:
        # test the image as a whole
        output = model(img_lq)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(tile, h, w)
        tile_overlap = tile_overlap
        sf = scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output

import torch.nn.functional as F

def pre_process(img, window_size):
    """
    Pad the input image to make its dimensions divisible by window_size.
    """
    _, _, h, w = img.size()
    mod_pad_h = (window_size - h % window_size) % window_size
    mod_pad_w = (window_size - w % window_size) % window_size
    img = F.pad(img, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return img, mod_pad_h, mod_pad_w

def post_process(output, mod_pad_h, mod_pad_w, scale):
    """
    Remove the padding added during preprocessing.
    """
    _, _, h, w = output.size()
    return output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

# def run(model, data_path, save_path, tile, device):
#     data_range = 1.0
#     sf = 4
#     border = sf

#     if data_path.endswith('/'):  # solve when path ends with /
#         data_path = data_path[:-1]
#     # scan all the jpg and png images
#     input_img_list = sorted(glob.glob(os.path.join(data_path, '*.[jpJP][pnPN]*[gG]')))
#     # save_path = os.path.join(args.save_dir, model_name, mode)
#     util.mkdir(save_path)

#     for i, img_lr in enumerate(input_img_list):

#         # --------------------------------
#         # (1) img_lr
#         # --------------------------------
#         img_name, ext = os.path.splitext(os.path.basename(img_lr))
#         print(img_name)
#         img_lr = util.imread_uint(img_lr, n_channels=3)
#         img_lr = util.uint2tensor4(img_lr, data_range)
#         img_lr = img_lr.to(device)

#         # --------------------------------
#         # (2) img_sr
#         # --------------------------------
#         img_sr = forward(img_lr, model, tile)
#         img_sr = util.tensor2uint(img_sr, data_range)

#         util.imsave(img_sr, os.path.join(save_path, img_name+ext))

def run(model, data_path, save_path, tile, device):
    data_range = 1.0
    sf = 4
    border = sf
    window_size = 16  # Set this to your model's window_size

    if data_path.endswith('/'):  # solve when path ends with /
        data_path = data_path[:-1]
    # scan all the jpg and png images
    input_img_list = sorted(glob.glob(os.path.join(data_path, '*.[jpJP][pnPN]*[gG]')))
    # save_path = os.path.join(args.save_dir, model_name, mode)
    util.mkdir(save_path)

    for i, img_lr in enumerate(input_img_list):
        print(f"Processing image {i+1}/{len(input_img_list)}: {img_lr}")

        # --------------------------------
        # (1) img_lr
        # --------------------------------
        img_name, ext = os.path.splitext(os.path.basename(img_lr))
        img_lr = util.imread_uint(img_lr, n_channels=3)
        img_lr = util.uint2tensor4(img_lr, data_range)
        img_lr = img_lr.to(device)

        # Pad the input image
        img_lr, mod_pad_h, mod_pad_w = pre_process(img_lr, window_size)

        # --------------------------------
        # (2) img_sr
        # --------------------------------
        img_sr = forward(img_lr, model, tile)
        img_sr = post_process(img_sr, mod_pad_h, mod_pad_w, sf)
        img_sr = util.tensor2uint(img_sr, data_range)

        # Save the output image
        output_image_path = os.path.join(save_path, img_name+ext)
        util.imsave(img_sr, output_image_path)
        print(f"Saved output image: {output_image_path}")

def main(model_dir, input_path, output_path, device=None):
    utils_logger.logger_info("NTIRE2024-ImageSRx4", log_path="NTIRE2024-ImageSRx4.log")
    logger = logging.getLogger("NTIRE2024-ImageSRx4")

    # --------------------------------
    # basic settings
    # --------------------------------
    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    if device is None:
        # device='cpu'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Running on device: {device}')

    json_dir = os.path.join(os.getcwd(), "results.json")
    if not os.path.exists(json_dir):
        results = dict()
    else:
        with open(json_dir, "r") as f:
            results = json.load(f)

    # --------------------------------
    # load model
    # --------------------------------
    # DAT baseline, ICCV 2023
    
    model = HAT()
    checkpoint = torch.load(model_dir)
    state_dict = checkpoint["params_ema"]
    model.load_state_dict(state_dict, strict=True)
    # model.load_state_dict(torch.load(model_dir), strict=True)
    print('HERE ')
    print('-------------------------------------------------------')
    model.eval()
    tile = None
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    
    run(model, input_path, output_path, tile, device)