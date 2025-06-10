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
import importlib

from sklearn.cluster import KMeans
import numpy as np

def fuse_outputs(img_srs):
    K=3
    img_srs_tmp = []
    for img_sr in img_srs:
        img_srs_tmp.append(img_sr.flatten().cpu().numpy())

    img_srs = img_srs_tmp
    # K-means clustering
    kmeans = KMeans(n_clusters=K, random_state=0, n_init=10).fit(np.array(img_srs))
    cluster_centers = kmeans.cluster_centers_

    # Assign weights based on cluster size
    weights = [len(np.where(kmeans.labels_ == i)[0]) for i in range(K)]
    weights = [w / sum(weights) for w in weights]

    # Output fusion
    img_sr_fused = np.zeros_like(cluster_centers[0])
    for center, weight in zip(cluster_centers, weights):
        img_sr_fused += center * weight

    img_sr_fused = torch.from_numpy(img_sr_fused.reshape(img_sr.shape)).to(img_sr.device)
    return img_sr_fused

def forward(img_lq, models, tile=None, tile_overlap=32, scale=4):
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

                out_patches = [model(in_patch) for model in models]
                out_patch = fuse_outputs(out_patches)

                # out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output

def run(models, data_path, save_path, tile, device):
    data_range = 1.0
    sf = 4
    border = sf

    if data_path.endswith('/'):  # solve when path ends with /
        data_path = data_path[:-1]
    # scan all the jpg and png images
    input_img_list = sorted(glob.glob(os.path.join(data_path, '*.[jpJP][pnPN]*[gG]')))
    # save_path = os.path.join(args.save_dir, model_name, mode)
    # import pdb; pdb.set_trace()
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
        img_sr = forward(img_lr, models, tile)
        img_sr = util.tensor2uint(img_sr, data_range)

        util.imsave(img_sr, os.path.join(save_path, img_name+ext))


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
    model1_module = importlib.import_module(f'models.team18_HIM.model_1')
    model2_module = importlib.import_module(f'models.team18_HIM.model_2')
    model3_module = importlib.import_module(f'models.team18_HIM.model_3')

    model1_path = os.path.join(model_dir, 'iqcmix_l2.pth')
    model2_path = os.path.join(model_dir, 'hatl_l2_sz512_2.pth')
    model3_path = os.path.join(model_dir, 'hatl_l2_sz512.pth')
    model4_path = os.path.join(model_dir, 'hatm_l2_sz384.pth')
    model5_path = os.path.join(model_dir, 'hatm_l2_sz512.pth')

    model1 = getattr(model1_module, f'HATIQCMix')(
                img_size=64,
                patch_size=1,
                in_chans=3,
                embed_dim=180,
                depths=(6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
                num_heads=(6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
                window_size=16,
                compress_ratio=3,
                squeeze_factor=30,
                conv_scale=0.01,
                overlap_ratio=0.5,
                mlp_ratio=2.,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.1,
                # norm_layer=nn.LayerNorm,
                ape=False,
                patch_norm=True,
                use_checkpoint=False,
                upscale=4,
                img_range=1.,
                upsampler='pixelshuffle',
                resi_connection='1conv')
    model1.load_state_dict(torch.load(model1_path)['params_ema'], strict=True)

    model2 = getattr(model2_module, f'HAT')(
                img_size=64,
                patch_size=1,
                in_chans=3,
                embed_dim=180,
                depths=(6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
                num_heads=(6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
                window_size=16,
                compress_ratio=3,
                squeeze_factor=30,
                conv_scale=0.01,
                overlap_ratio=0.5,
                mlp_ratio=2.,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.1,
                # norm_layer=nn.LayerNorm,
                ape=False,
                patch_norm=True,
                use_checkpoint=False,
                upscale=4,
                img_range=1.,
                upsampler='pixelshuffle',
                resi_connection='1conv')
    model2.load_state_dict(torch.load(model2_path)['params_ema'], strict=True)

    model3 = getattr(model2_module, f'HAT')(
                img_size=64,
                patch_size=1,
                in_chans=3,
                embed_dim=180,
                depths=(6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
                num_heads=(6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
                window_size=16,
                compress_ratio=3,
                squeeze_factor=30,
                conv_scale=0.01,
                overlap_ratio=0.5,
                mlp_ratio=2.,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.1,
                # norm_layer=nn.LayerNorm,
                ape=False,
                patch_norm=True,
                use_checkpoint=False,
                upscale=4,
                img_range=1.,
                upsampler='pixelshuffle',
                resi_connection='1conv')
    model3.load_state_dict(torch.load(model3_path)['params_ema'], strict=True)

    model4 = getattr(model3_module, f'HATM')(
                img_size=64,
                patch_size=1,
                in_chans=3,
                embed_dim=180,
                depths=(6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
                num_heads=(6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
                window_size=16,
                compress_ratio=3,
                squeeze_factor=30,
                conv_scale=0.01,
                overlap_ratio=0.5,
                mlp_ratio=2.,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.1,
                # norm_layer=nn.LayerNorm,
                ape=False,
                patch_norm=True,
                use_checkpoint=False,
                upscale=4,
                img_range=1.,
                upsampler='pixelshuffle',
                resi_connection='1conv')
    model4.load_state_dict(torch.load(model4_path)['params_ema'], strict=True)

    model5 = getattr(model3_module, f'HATM')(
                img_size=64,
                patch_size=1,
                in_chans=3,
                embed_dim=180,
                depths=(6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
                num_heads=(6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
                window_size=16,
                compress_ratio=3,
                squeeze_factor=30,
                conv_scale=0.01,
                overlap_ratio=0.5,
                mlp_ratio=2.,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.1,
                # norm_layer=nn.LayerNorm,
                ape=False,
                patch_norm=True,
                use_checkpoint=False,
                upscale=4,
                img_range=1.,
                upsampler='pixelshuffle',
                resi_connection='1conv')
    model5.load_state_dict(torch.load(model5_path)['params_ema'], strict=True)


    models = [model1.eval(), model2.eval(), model3.eval(), model4.eval(), model5.eval()]

    tile = 192
    for model in models:
        for k, v in model.named_parameters():
            v.requires_grad = False
        model = model.to(device)
    
    run(models, input_path, output_path, tile, device)