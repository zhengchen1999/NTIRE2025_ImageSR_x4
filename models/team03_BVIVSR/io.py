import argparse
import os
from PIL import Image
import torch
from torchvision import transforms
from .mymodels import *
from collections import OrderedDict

from pprint import pprint
from utils.model_summary import get_model_flops
from utils import utils_logger
from utils import utils_image as util
import os.path
import logging
import json
import glob

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def bicubic_resample(image, scale):
    return image.resize((int(image.width * scale), int(image.height * scale)), Image.BICUBIC)

def batched_predict_fast(model, inp, coord, cell, bsize=300):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell)
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=2)
    return pred

def process_image(img_lr, model, scale, scale_max=4):
    img_lr = img_lr.squeeze(0)
    window_size = 64

    h = int(img_lr.shape[-2] * scale)
    w = int(img_lr.shape[-1] * scale)

    coord = make_coord((h, w)).cuda()
    cell = torch.tensor([2 / h, 2 / w], dtype=torch.float32).unsqueeze(0)

    inp = (img_lr - 0.5) / 0.5

    if window_size != 0:
        _, h_old, w_old = inp.size()
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        inp = torch.cat([inp, torch.flip(inp, [1])], 1)[:, :h_old + h_pad, :]
        inp = torch.cat([inp, torch.flip(inp, [2])], 2)[:, :, :w_old + w_pad]

        coord = make_coord((round(scale * (h_old + h_pad)), round(scale * (w_old + w_pad))), flatten=False).unsqueeze(0).cuda()
        cell = torch.ones_like(cell)
        cell[:, 0] *= 2 / inp.shape[-2] / scale
        cell[:, 1] *= 2 / inp.shape[-1] / scale


    cell_factor = max(scale / scale_max, 1)
    pred = batched_predict_fast(model, inp.unsqueeze(0).cuda(), coord, cell_factor * cell.cuda(), bsize=300)[0]

    pred = (pred * 0.5 + 0.5).clamp(0, 1)

    shape = [3, round(scale * (h_old + h_pad)), round(scale * (w_old + w_pad))]
    pred = pred.view(*shape).contiguous()
    output = pred[..., :h, :w]

    return output.unsqueeze(0)



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
        img_sr = process_image(img_lr, model, sf)
        img_sr = util.tensor2uint(img_sr, data_range)

        util.imsave(img_sr, os.path.join(save_path, img_name + ext))


def main(model_dir, input_path, output_path, device=None):
    utils_logger.logger_info("NTIRE2024-ImageSRx4", log_path="NTIRE2024-ImageSRx4.log")
    logger = logging.getLogger("NTIRE2024-ImageSRx4")

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

    state_dict = torch.load(model_dir, map_location="cuda:0")['model']
    new_sd = OrderedDict()
    for key, value in state_dict['sd'].items():
        new_key = key.replace('module.', '')
        new_sd[new_key] = value
    state_dict['sd'] = new_sd

    model = make(state_dict, load_sd=True).cuda()

    model.eval()
    tile = None
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    run(model, input_path, output_path, tile, device)