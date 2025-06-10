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

# from models.team00_DAT.model import DAT
from models.team01_PMELSR.basicsr.archs.mambairv2_arch import MambaIRv2
from models.team01_PMELSR.basicsr.archs.dat_arch import DAT
from models.team01_PMELSR.basicsr.archs.hat_arch import HATLocal
from models.team01_PMELSR.basicsr.archs.rrdbnet_arch import RRDBNet

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

        print("name: {}".format(os.path.basename(img_lr)))
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


def model_ensemble(dir3):
    import cv2
    import numpy as np

    dir1 = dir3.replace('01_MambaIRv2','01_DAT')
    dir2 = dir3.replace('01_MambaIRv2','01_HAT')
    # dir3 = r"results/01_MambaIRv2/test"

    weights = [0.02, 0.22, 0.76]

    output_dir = dir3.replace('01_MambaIRv2','01_Model_Ensemble')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("weights: {}".format(weights))

    img_path = glob.glob(dir1 + "/*.png")

    for p in img_path:
        name = os.path.basename(p)

        path1 = p
        path2 = os.path.join(dir2, name)
        path3 = os.path.join(dir3, name)

        img1 = cv2.imread(path1, cv2.IMREAD_COLOR).astype(np.float32)
        img2 = cv2.imread(path2, cv2.IMREAD_COLOR).astype(np.float32)
        img3 = cv2.imread(path3, cv2.IMREAD_COLOR).astype(np.float32)

        img_mean = img1 * weights[0] + img2 * weights[1] + img3 * weights[2]

        img_mean = img_mean.round().astype(np.uint8)

        cv2.imwrite(os.path.join(output_dir, name), img_mean)


def all_remove(dir):
    import shutil
    shutil.rmtree(dir)
    os.removedirs(dir.replace('test', '').replace('valid', ''))
    # list = os.listdir(dir)
    # for l in list:
    #     if l.endswith(ends):
    #         os.remove(os.path.join(dir, l))
    # print("删除成功")


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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Running on device: {device}')

    json_dir = os.path.join(os.getcwd(), "results.json")
    if not os.path.exists(json_dir):
        results = dict()
    else:
        with open(json_dir, "r") as f:
            results = json.load(f)

    model_dir = model_dir.replace('PMELSR_x4.pth', 'MambaIRv2_x4.pth')
    output_path = output_path.replace('01_PMELSR', '01_MambaIRv2')

    # --------------------------------
    # load model
    # --------------------------------

    # MambaIRv2
    print("Using MambaIRv2 for Inference")
    model = MambaIRv2(
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=174, # 48
                 d_state=16, # 8
                 depths=(6, 6, 6, 6, 6, 6, 6, 6, 6), # (6, 6, 6, 6,),
                 num_heads=(6, 6, 6, 6, 6, 6, 6, 6, 6), #(4, 4, 4, 4,),
                 window_size=16,
                 inner_rank=64, #32
                 num_tokens=128, #64
                 convffn_kernel_size=5,
                 mlp_ratio=2.,
                 upscale=4,
                 upsampler='pixelshuffle',
                 resi_connection='1conv')
    # model.load_state_dict(torch.load(model_dir), strict=True)

    loadnet = torch.load(model_dir)
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'
    model.load_state_dict(loadnet[keyname], strict=True)
   
    model.eval()
    tile = None
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    
    run(model, input_path, output_path, tile, device)

    # DAT
    print("Using DAT for Inference")
    model = DAT(
        upscale=4,
        in_chans=3,
        img_size=64,
        img_range=1.,
        split_size=[8, 32],
        depth=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        expansion_factor=4,
        resi_connection='1conv',
    )
    # model.load_state_dict(torch.load(model_dir), strict=True)

    loadnet = torch.load(model_dir.replace('MambaIRv2_x4.pth', 'DAT_x4.pth'))
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'
    model.load_state_dict(loadnet[keyname], strict=True)

    model.eval()
    tile = None
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    run(model, input_path, output_path.replace('01_MambaIRv2','01_DAT'), tile, device)

    # HAT
    print("Using HAT for Inference")
    model = HATLocal(
        img_size=64,  # √
        patch_size=1,
        in_chans=3,  # √
        embed_dim=180,  # √
        depths=(6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),  # √
        num_heads=(6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),  # √
        window_size=16,  # √
        compress_ratio=3,  # √
        squeeze_factor=30,  # √
        conv_scale=0.01,  # √
        overlap_ratio=0.5,  # √
        mlp_ratio=2,  # √
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        upscale=4,  # √
        img_range=1.,  # √
        upsampler='pixelshuffle',  # √
        resi_connection='1conv')  # √
    # model.load_state_dict(torch.load(model_dir), strict=True)

    loadnet = torch.load(model_dir.replace('MambaIRv2_x4.pth', 'HAT_x4.pth'))
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'
    model.load_state_dict(loadnet[keyname], strict=True)

    model.eval()
    tile = None
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    run(model, input_path, output_path.replace('01_MambaIRv2','01_HAT'), tile, device)

    # ensemble
    print("Using Model_Ensemble for Inference")
    model_ensemble(output_path)

    # RRDB
    print("Using RRDB for Inference")
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=1)
    # model.load_state_dict(torch.load(model_dir), strict=True)

    loadnet = torch.load(model_dir.replace('MambaIRv2_x4.pth', 'RRDBNet_x4.pth'))
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'
    model.load_state_dict(loadnet[keyname], strict=True)

    model.eval()
    tile = None
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    run(model, output_path.replace('01_MambaIRv2','01_Model_Ensemble'), output_path.replace('01_MambaIRv2','01_PMELSR'), tile, device)

    all_remove(output_path.replace('01_MambaIRv2','01_MambaIRv2'))
    all_remove(output_path.replace('01_MambaIRv2', '01_HAT'))
    all_remove(output_path.replace('01_MambaIRv2','01_DAT'))
    all_remove(output_path.replace('01_MambaIRv2','01_Model_Ensemble'))



