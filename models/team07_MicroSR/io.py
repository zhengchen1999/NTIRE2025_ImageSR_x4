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
from models.team07_MicroSR.model import MicroSR
from tqdm import tqdm

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

    window_size = 16
    for i, img_lr in tqdm(enumerate(input_img_list), total=len(input_img_list)):

        # --------------------------------
        # (1) img_lr
        # --------------------------------
        img_name, ext = os.path.splitext(os.path.basename(img_lr))
        img_lr = util.imread_uint(img_lr, n_channels=3)


        img_lr = util.uint2tensor4(img_lr, data_range)
        img_lr = img_lr.to(device)
        try:
            with torch.no_grad():
                #output = model(img)
                _, _, h_old, w_old = img_lr.size()
                h_pad = (h_old // window_size + 1) * window_size - h_old
                w_pad = (w_old // window_size + 1) * window_size - w_old
                img_lr = torch.cat([img_lr, torch.flip(img_lr, [2])], 2)[:, :, :h_old + h_pad, :]
                img_lr = torch.cat([img_lr, torch.flip(img_lr, [3])], 3)[:, :, :, :w_old + w_pad]
                img_sr = forward(img_lr, model, tile)
                img_sr = img_sr[..., :h_old * sf, :w_old * sf]

                # --------------------------------
                # (2) img_sr
                # --------------------------------

                img_sr = util.tensor2uint(img_sr, data_range)

                util.imsave(img_sr, os.path.join(save_path, img_name+ext))

        except Exception as error:
            print('Error', error, img_lr)



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

    # --------------------------------
    # load model
    # --------------------------------
    
    # model = MicroSR(upscale=4, in_chans=3,  img_size= 64, window_size= 16, compress_ratio= 3,squeeze_factor= 30,
    #                     conv_scale= 0.01, overlap_ratio= 0.5, img_range= 1., depths= [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    #                     embed_dim= 180, num_heads= [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], gc= 32,
    #                     mlp_ratio= 2, upsampler= 'pixelshuffle', resi_connection= '1conv')
    model = MicroSR(upscale=4, in_chans=3,  img_size= 64, window_size= 16, compress_ratio= 3,squeeze_factor= 30,
                    conv_scale= 0.01, overlap_ratio= 0.5, img_range= 1., depths= [6, 6, 6, 6, 6, 6],
                    embed_dim= 180, num_heads= [6, 6, 6, 6, 6, 6], gc= 32,
                    mlp_ratio= 2, upsampler= 'pixelshuffle', resi_connection= '1conv')  

    model.load_state_dict(torch.load(model_dir)['params'], strict=True)
   
    model.eval()
    tile = None
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    
    run(model, input_path, output_path, tile, device)