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

from models.team10_ATM.model import ATM

import torch.nn.functional as F

def tt(x, model, device, pz=220, scale=4):
    _, C, h, w = x.shape
    split_token_h = h // pz + 1  # number of horizontal cut sections
    split_token_w = w // pz + 1 # number of vertical cut sections
    # padding
    mod_pad_h, mod_pad_w = 0, 0
    if h % split_token_h != 0:
        mod_pad_h = split_token_h - h % split_token_h
    if w % split_token_w != 0:
        mod_pad_w = split_token_w - w % split_token_w
    img = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect').to(device)
    _, _, H, W = img.size()
    split_h = H // split_token_h  # height of each partition
    split_w = W // split_token_w  # width of each partition
    # overlapping
    shave_h = split_h // 10
    shave_w = split_w // 10
    ral = H // split_h
    row = W // split_w
    slices = []  # list of partition borders
    for i in range(ral):
        for j in range(row):
            if i == 0 and i == ral - 1:
                top = slice(i * split_h, (i + 1) * split_h)
            elif i == 0:
                top = slice(i*split_h, (i+1)*split_h+shave_h)
            elif i == ral - 1:
                top = slice(i*split_h-shave_h, (i+1)*split_h)
            else:
                top = slice(i*split_h-shave_h, (i+1)*split_h+shave_h)
            if j == 0 and j == row - 1:
                left = slice(j*split_w, (j+1)*split_w)
            elif j == 0:
                left = slice(j*split_w, (j+1)*split_w+shave_w)
            elif j == row - 1:
                left = slice(j*split_w-shave_w, (j+1)*split_w)
            else:
                left = slice(j*split_w-shave_w, (j+1)*split_w+shave_w)
            temp = (top, left)
            slices.append(temp)
    img_chops = []  # list of partitions
    for temp in slices:
        top, left = temp
        img_chops.append(img[..., top, left])
    
 
    with torch.inference_mode():
        outputs = []
        for chop in img_chops:
            out = model(chop)  # image processing of each partition
            outputs.append(out)
        _img = torch.zeros(1, C, H * scale, W * scale)
        # merge
        for i in range(ral):
            for j in range(row):
                top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                left = slice(j * split_w * scale, (j + 1) * split_w * scale)
                if i == 0:
                    _top = slice(0, split_h * scale)
                else:
                    _top = slice(shave_h * scale, (shave_h + split_h) * scale)
                if j == 0:
                    _left = slice(0, split_w * scale)
                else:
                    _left = slice(shave_w * scale, (shave_w + split_w) * scale)
                _img[..., top, left] = outputs[i * row + j][..., _top, _left]
        output = _img
        
    _, _, h, w = output.size()
    output = output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale].to(device)

    return output

def forward(lq, model, device, scale=4):
    
    def _transform(v, op):
        v2np = v.detach().cpu().numpy()
        if op == 'v':
            tfnp = v2np[:, :, :, ::-1].copy()
        elif op == 'h':
            tfnp = v2np[:, :, ::-1, :].copy()
        elif op == 't':
            tfnp = v2np.transpose((0, 1, 3, 2)).copy()

        ret = torch.Tensor(tfnp).to(device)
        return ret

      
    lq_list = [lq]
    for tf in 'v', 'h', 't':
        lq_list.extend([_transform(t, tf) for t in lq_list])

    # inference
    out_list = [tt(aug,model, device,scale=scale) for aug in lq_list]


    # merge results
    for i in range(len(out_list)):
        if i > 3:
            out_list[i] = _transform(out_list[i], 't')
        if i % 4 > 1:
            out_list[i] = _transform(out_list[i], 'h')
        if (i % 4) % 2 == 1:
            out_list[i] = _transform(out_list[i], 'v')
    output = torch.cat(out_list, dim=0)

    return output.mean(dim=0, keepdim=True)


def run(model, data_path, save_path, device):
    data_range = 1.0
    # sf = 4
    # border = sf

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
        img_sr = forward(img_lr, model,device)
        img_sr = util.tensor2uint(img_sr, data_range)

        util.imsave(img_sr, os.path.join(save_path, img_name+ext))


def main(model_dir, input_path, output_path, device=None):
    # utils_logger.logger_info("NTIRE2025-ImageSRx4", log_path="NTIRE2025-ImageSRx4.log")
    # logger = logging.getLogger("NTIRE2025-ImageSRx4")

    # --------------------------------
    # basic settings
    # --------------------------------
    # torch.cuda.current_device()
    # torch.cuda.empty_cache()
    # torch.backends.cudnn.benchmark = False
    # if device is None:
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     print(f'Running on device: {device}')

    # json_dir = os.path.join(os.getcwd(), "results.json")
    # if not os.path.exists(json_dir):
    #     results = dict()
    # else:
    #     with open(json_dir, "r") as f:
    #         results = json.load(f)

    # --------------------------------
    # load model
    # --------------------------------
    
    
    model = ATM(
        upscale=4,
        img_size=96,
        embed_dim=180,
        depths=[6, 6, 6, 6, 6, 6, ],
        num_heads=[6, 6, 6, 6, 6, 6, ],
        window_size=16,
        category_size=128,
        num_tokens=128,
        reducted_dim=20,
        convffn_kernel_size=5,
        img_range=1.,
        mlp_ratio=2,
        upsampler='pixelshuffle')
    model.load_state_dict(torch.load(model_dir)['params_ema'], strict=True)
   
    model.eval()
    # tile = None
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    
    run(model, input_path, output_path, device)