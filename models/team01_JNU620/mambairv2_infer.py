# Modified from https://github.com/JingyunLiang/SwinIR
# 需要实现测试时数据增广(test time augmentation,TTA)--trick
import argparse
import cv2
import glob
import numpy as np
import os
import torch
from torch.nn import functional as F
import torch.nn as nn

from models.team01_PMELSR.basicsr.archs.mambairv2_arch import MambaIRv2

"""
python infer.py
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='datasets/DIV2K/DIV2K_valid_LR_bicubic_X4', help='input test image folder')
    parser.add_argument('--output', type=str, default='results/infer/01mytta', help='output folder')
    parser.add_argument(
        '--task',
        type=str,
        default='classical_sr',
        help='classical_sr, lightweight_sr, real_sr, gray_dn, color_dn, jpeg_car')
    # dn: denoising; car: compression artifact removal
    # TODO: it now only supports sr, need to adapt to dn and jpeg_car
    parser.add_argument('--patch_size', type=int, default=1, help='training patch size')
    # add by wedream.
    parser.add_argument('--window_size', type=int, default=16, help='window size')
    # add by wedream. use TTA trick.
    parser.add_argument("--tta", default=True, action="store_false", help="use TTA or not")
    parser.add_argument("--alpha", type=float, default=0.3, help="the coefficient of sr")
    parser.add_argument("--beta", type=float, default=0.5, help="the coefficient of horizontal sr")
    parser.add_argument("--gama", type=float, default=0.2, help="the coefficient of rotate sr")
    # end.
    parser.add_argument('--scale', type=int, default=4, help='scale factor: 1, 2, 3, 4, 8')  # 1 for dn and jpeg car
    parser.add_argument('--noise', type=int, default=15, help='noise level: 15, 25, 50')
    parser.add_argument('--jpeg', type=int, default=40, help='scale factor: 10, 20, 30, 40')
    parser.add_argument('--large_model', action='store_true', help='Use large model, only used for real image sr')
    parser.add_argument(
        '--model_path',
        type=str,
        default='experiments/pretrained_models/mambairv2_classicSR_Large_x4.pth')
    args = parser.parse_args()

    print("#"*50)
    print(args)
    print("#" * 50)

    os.makedirs(args.output, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = define_model(args)
    model.eval()
    model = model.to(device)


    window_size = args.window_size

    for idx, path in enumerate(sorted(glob.glob(os.path.join(args.input, '*')))):
        # read image
        imgname = os.path.splitext(os.path.basename(path))[0]
        print('Testing', idx, imgname)
        # read image
        img_origin = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        # img = torch.from_numpy(np.transpose(img_origin[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = torch.from_numpy(np.transpose(img_origin[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(device)


        # test time augmentation. add by wedream.
        if args.tta:
            # horizontal
            img_h = cv2.flip(img_origin, 1)
            img_h = torch.from_numpy(np.transpose(img_h[:, :, [2, 1, 0]], (2, 0, 1))).float()
            img_h = img_h.unsqueeze(0).to(device)
            # rotate, 逆时针旋转90°
            img_r = img_origin.transpose(1, 0, 2)
            img_r = torch.from_numpy(np.transpose(img_r[:, :, [2, 1, 0]], (2, 0, 1))).float()
            img_r = img_r.unsqueeze(0).to(device)

        # end.

        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            mod_pad_h, mod_pad_w = 0, 0
            _, _, h, w = img.size()
            if h % window_size != 0:
                mod_pad_h = window_size - h % window_size
            if w % window_size != 0:
                mod_pad_w = window_size - w % window_size
            img = F.pad(img, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

            output = model(img)
            _, _, h, w = output.size()
            output = output[:, :, 0:h - mod_pad_h * args.scale, 0:w - mod_pad_w * args.scale]


            if args.tta:
                # pad input image to be a multiple of window_size
                mod_pad_h, mod_pad_w = 0, 0
                _, _, h, w = img_h.size()
                if h % window_size != 0:
                    mod_pad_h = window_size - h % window_size
                if w % window_size != 0:
                    mod_pad_w = window_size - w % window_size
                img_h = F.pad(img_h, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

                output_h = model(img_h)
                _, _, h, w = output_h.size()
                output_h = output_h[:, :, 0:h - mod_pad_h * args.scale, 0:w - mod_pad_w * args.scale]

                # pad input image to be a multiple of window_size
                mod_pad_h, mod_pad_w = 0, 0
                _, _, h, w = img_r.size()
                if h % window_size != 0:
                    mod_pad_h = window_size - h % window_size
                if w % window_size != 0:
                    mod_pad_w = window_size - w % window_size
                img_r = F.pad(img_r, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

                output_r = model(img_r)
                _, _, h, w = output_r.size()
                output_r = output_r[:, :, 0:h - mod_pad_h * args.scale, 0:w - mod_pad_w * args.scale]


        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        # cv2.imwrite(os.path.join(args.output, f'{imgname}_origin.png'), output)

        if args.tta:
            # save image
            output_h = output_h.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            if output_h.ndim == 3:
                output_h = np.transpose(output_h[[2, 1, 0], :, :], (1, 2, 0))
            output_h = (output_h * 255.0).round().astype(np.uint8)

            # 水平翻转，再翻转就是原图
            output_h = cv2.flip(output_h, 1)

            # cv2.imwrite(os.path.join(args.output, f'{imgname}_h.png'), output_h)

            # save image
            output_r = output_r.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            if output_r.ndim == 3:
                output_r = np.transpose(output_r[[2, 1, 0], :, :], (1, 2, 0))
            output_r = (output_r * 255.0).round().astype(np.uint8)

            # 转置90°,再转置90°,就是原图
            output_r = output_r.transpose(1, 0, 2)

            # cv2.imwrite(os.path.join(args.output, f'{imgname}_r.png'), output_r)

            output = output.astype(np.float64)
            output_h = output_h.astype(np.float64)
            output_r = output_r.astype(np.float64)

            # 建议添加权重来计算
            # output_tta = output*0.4+output_h*0.4+output_r*0.2
            # output_tta = output * args.alpha + output_h * args.beta + output_r * args.gama
            output_tta = (output + output_h + output_r )/3

            cv2.imwrite(os.path.join(args.output, f'{imgname}_tta.png'), output_tta)


        else :
            cv2.imwrite(os.path.join(args.output, f'{imgname}_origin.png'), output)


def define_model(args):
    # 001 classical image sr
    model = None
    if args.task == 'classical_sr':
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

    loadnet = torch.load(args.model_path)
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'
    model.load_state_dict(loadnet[keyname], strict=True)

    return model


if __name__ == '__main__':
    # main()
    # hat = HAT()
    # print(hat)
    main()