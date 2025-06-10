import argparse
import cv2
import glob
import numpy as np
import os
import torch
from tqdm import tqdm
from torch.multiprocessing import Process, set_start_method


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

from models.team15_SMT_HAT.MSHAT_model import MSHAT

import torch.multiprocessing as mp
import resource


torch.multiprocessing.set_sharing_strategy('file_system')


soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard_limit), hard_limit))

import time

def process_img(img_t,model):
    b, c, h_old, w_old = img_t.size()
    h_pad = (h_old // 16 + 1) * 16 - h_old
    w_pad = (w_old // 16 + 1) * 16 - w_old
    img_t_padded = torch.cat([img_t, torch.flip(img_t, [2])], 2)[:, :, :h_old + h_pad, :]
    img_t_padded = torch.cat([img_t_padded, torch.flip(img_t_padded, [3])], 3)[:, :, :, :w_old + w_pad]

    output = test(img_t_padded,model)
    output = output[..., :h_old * 4, :w_old * 4]

    return output

def process_images_on_gpu(gpu_id, image_paths_chunk, model, save_path):
    """Process a chunk of images on a specific GPU."""
    device = torch.device(f'cuda:{gpu_id}')
    model = model.to(device)
    model.eval()

    # Define TTA transformations and their inverses
    transforms = [
        lambda x: x,                                           # Original
        lambda x: x.flip(-1),                                  # Horizontal flip
        lambda x: x.flip(-2),                                  # Vertical flip
        lambda x: x.flip(-1).flip(-2),                         # Horizontal + Vertical flip
        lambda x: x.permute(0, 1, 3, 2),                       # Transpose
        lambda x: x.permute(0, 1, 3, 2).flip(-1),              # Transpose + Horizontal flip
        lambda x: x.permute(0, 1, 3, 2).flip(-2),              # Transpose + Vertical flip
        lambda x: x.permute(0, 1, 3, 2).flip(-1).flip(-2),     # Transpose + Horizontal + Vertical flip
    ]

    inv_transforms = [
        lambda x: x,                                           # Original
        lambda x: x.flip(-1),                                  # Horizontal flip
        lambda x: x.flip(-2),                                  # Vertical flip
        lambda x: x.flip(-1).flip(-2),                         # Horizontal + Vertical flip
        lambda x: x.permute(0, 1, 3, 2),                       # Transpose
        lambda x: x.flip(-1).permute(0, 1, 3, 2),              # Horizontal flip + Transpose
        lambda x: x.flip(-2).permute(0, 1, 3, 2),              # Vertical flip + Transpose
        lambda x: x.flip(-1).flip(-2).permute(0, 1, 3, 2),     # Horizontal + Vertical flip + Transpose
    ]
    
    for path in tqdm(image_paths_chunk, desc=f'GPU {gpu_id} Testing'):
        imgname = os.path.splitext(os.path.basename(path))[0]
        if os.path.isfile(os.path.join(save_path, f'{imgname}.png')):
            continue
        
        img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float().unsqueeze(0).to(device)

        outputs_tta = []
        with torch.no_grad():
            for t_func, inv_func in zip(transforms, inv_transforms):
                img_t=t_func(img)

                output=process_img(img_t,model)

                inv_trans_img=inv_func(output)
                outputs_tta.append(inv_trans_img)


        final_output = torch.mean(torch.stack(outputs_tta), dim=0)
        
        final_output = final_output.clamp(0, 1).cpu().numpy()
        out_img = np.transpose(final_output[0][[2, 1, 0], :, :], (1, 2, 0))
        out_img = (out_img * 255.0).round().astype(np.uint8)
        cv2.imwrite(os.path.join(save_path, f'{imgname}.png'), out_img)


def inference_TTA(model, data_path, save_path, tile, device):
    """Performs inference with TTA using multiple GPUs."""
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPU devices found.")
    print(f"Using {num_gpus} GPUs for inference.")
    os.makedirs(save_path, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(data_path, '*.[jpJP][pnPN]*[gG]')))
    image_chunks = [image_paths[i::num_gpus] for i in range(num_gpus)]
    
    processes = []
    for gpu_id, image_paths_chunk in enumerate(image_chunks):
        p = Process(target=process_images_on_gpu, args=(gpu_id, image_paths_chunk, model, save_path))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


def test(img_lq, model, tile=None, tile_overlap=32, scale=4):
    if tile is None:
        # test the image as a whole
        output = model(img_lq)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(tile, h, w)
        assert tile % 16 == 0, "tile size should be a multiple of window_size"
        tile_overlap = tile_overlap
        sf = 4

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


def main(model_dir, input_path, output_path, device=None): 
    utils_logger.logger_info("NTIRE2025-ImageSRx4", log_path="NTIRE2025-ImageSRx4.log")
    logger = logging.getLogger("NTIRE2025-ImageSRx4")
    
    
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
    
    try:
        set_start_method('spawn', force=True)
    except RuntimeError as e:
        print(f"RuntimeError: {e}")
        
        
        
    model=MSHAT(upscale=4, in_chans=3, img_size=64, window_size=16,compress_ratio= 3,squeeze_factor= 30,conv_scale= 0.01, overlap_ratio= 0.5,
                   img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                   mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
    
    model.load_state_dict(torch.load(model_dir)['params'], strict=False)


    tile = None
    inference_TTA(model, input_path, output_path, tile, device)