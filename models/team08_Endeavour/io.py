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

from models.team08_HAT_F.model import HAT
import torch.nn.functional as F
import torchvision.transforms as T
import torch



def forward(img_lq, model, tile=None, tile_overlap=32, scale=4, window_size=16):
    if tile is None:
        def _test_pad(lq):
            # pad to a multiple of window_size
            mod_pad_h, mod_pad_w = 0, 0
            _, _, h, w = lq.size()
            if h % window_size != 0:
                mod_pad_h = window_size - h % window_size
            if w % window_size != 0:
                mod_pad_w = window_size - w % window_size
            img = lq
            img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h + mod_pad_h, :]
            img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w + mod_pad_w]

            return img, mod_pad_h, mod_pad_w

        model.eval()
        img_lq, mod_pad_h, mod_pad_w = _test_pad(img_lq)

        with torch.no_grad():

            output = model(img_lq)


            output += torch.rot90(model(torch.rot90(img_lq, 1, dims=[2,3])), 3, dims=[2,3])
            output += torch.rot90(model(torch.rot90(img_lq, 2, dims=[2,3])), 2, dims=[2,3])
            output += torch.rot90(model(torch.rot90(img_lq, 3, dims=[2,3])), 1, dims=[2,3])


            h_transform = T.RandomHorizontalFlip(p=1)
            output += h_transform(model(h_transform(img_lq)))
            output += h_transform(torch.rot90(model(torch.rot90(h_transform(img_lq), 1, dims=[2,3])), 3, dims=[2,3]))
            output += h_transform(torch.rot90(model(torch.rot90(h_transform(img_lq), 2, dims=[2,3])), 2, dims=[2,3]))
            output += h_transform(torch.rot90(model(torch.rot90(h_transform(img_lq), 3, dims=[2,3])), 1, dims=[2,3]))


            output /= 8


        _, _, h, w = output.size()
        output = output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

        return output

    else:

        b, c, h, w = img_lq.size()
        tile = min(tile, h, w)
        sf = scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
        w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
        E = torch.zeros(b, c, h * sf, w * sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(out_patch)
                W[..., h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(out_patch_mask)
        output = E.div_(W)

        return output



def run(model, data_path, save_path, tile, device):
    data_range = 1.0
    sf = 4
    border = sf

    if data_path.endswith('/'):  # solve when path ends with /
        data_path = data_path[:-1]
    # scan all the jpg and png images
    input_img_list = sorted(glob.glob(os.path.join(data_path, '*.[jpJP][pnPN]*[gG]')))
    # save_path = os.path.join(args.save_dir, model_name, mode)
    util.mkdir(save_path)

    for i, img_lr in enumerate(input_img_list):

        # --------------------------------
        # (1) img_lr
        # --------------------------------
        img_name, ext = os.path.splitext(os.path.basename(img_lr))
        img_lr = util.imread_uint(img_lr, n_channels=3)
        img_lr = util.uint2tensor4(img_lr, data_range)
        img_lr = img_lr.to(device)

        # --------------------------------
        # (2) img_sr
        # --------------------------------
        img_sr = forward(img_lr, model, tile)
        img_sr = util.tensor2uint(img_sr, data_range)

        util.imsave(img_sr, os.path.join(save_path, img_name+ext))


def main(model_dir, input_path, output_path, device=None):
    utils_logger.logger_info("NTIRE2025-ImageSRx4", log_path="NTIRE2025-ImageSRx4.log")
    logger = logging.getLogger("NTIRE2025-ImageSRx4")

    # --------------------------------
    # basic settings
    # --------------------------------
    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    if device is None:
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
    # model.load_state_dict(torch.load(model_dir), strict=True)

    state_dict = torch.load(model_dir)


    if 'params_ema' in state_dict:
        model.load_state_dict(state_dict['params_ema'], strict=True)
    else:
        model.load_state_dict(state_dict, strict=True)

    model.eval()
    tile = None
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    run(model, input_path, output_path, tile, device)