import cv2
from datetime import datetime
import logging
import math
import numpy as np
import os
import random
from shutil import get_terminal_size
import sys
import time
from thop import profile, clever_format
import torch
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn.functional as F


class CosineAnnealingLR_warmup(_LRScheduler):
    def __init__(self, args, optimizer, base_lr, last_epoch=-1, min_lr=1e-7):
        self.base_lr = base_lr
        self.min_lr = min_lr
        ####The duration of the warm-up
        self.w_iter = args.WARM_UP_ITER

        self.w_fac = args.WARM_UP_FACTOR
        self.T_period = args.T_PERIOD
        self.last_restart = 0
        self.T_max = self.T_period[0]
        assert args.MAX_ITER == self.T_period[-1], 'Illegal training period setting.'
        super(CosineAnnealingLR_warmup, self).__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        ### cosine lr
        if self.last_epoch - self.last_restart < self.w_iter:
            ratio = self.w_fac + (1 - self.w_fac) * (self.last_epoch - self.last_restart) / self.w_iter
            return [(self.base_lr - self.min_lr) * ratio + self.min_lr for group in self.optimizer.param_groups]

        ### warm up for a period time
        elif self.last_epoch in self.T_period:
            self.last_restart = self.last_epoch
            if self.last_epoch != self.T_period[-1]:
                self.T_max = self.T_period[self.T_period.index(self.last_epoch) + 1]
            return [self.min_lr for group in self.optimizer.param_groups]
        else:
            ratio = 1 / 2 * (1 + math.cos(
                (self.last_epoch - self.last_restart - self.w_iter) / (self.T_max - self.last_restart - self.w_iter) * math.pi))
            return [(self.base_lr - self.min_lr) * ratio + self.min_lr for group in self.optimizer.param_groups]


class SubsectionLR(_LRScheduler):
    def __init__(self,
                 optimizer,
                 update_iter=(0, ),
                 update_weights=(1, ),
                 last_epoch=-1):
        self.update_iter = update_iter
        self.update_weights = update_weights
        assert len(self.update_iter) == len(
            self.update_weights), 'restarts and their weights do not match.'
        super(SubsectionLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        #print('self.update_iter',self.update_iter)
        #print('self.update_weights',self.update_weights)
        if self.last_epoch in self.update_iter:
            weight = self.update_weights[self.update_iter.index(self.last_epoch)]
            return [
                group['initial_lr'] * weight
                for group in self.optimizer.param_groups
            ]
        else:
            index = len(self.update_iter)-1
            for i in range(len(self.update_iter)):
                if self.last_epoch<self.update_iter[i]:
                    index = i-1
                    break
            if index==-1:
                return [
                    group['initial_lr'] for group in self.optimizer.param_groups
                ]
            else:
                return [
                    group['initial_lr'] * self.update_weights[index] for group in self.optimizer.param_groups
                ]


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    # 4D: grid (B, C, H, W), 3D: (C, H, W), 2D: (H, W)
    # output_channel: bgr
    
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])

    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), padding=0, normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))

    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()

    return img_np.astype(out_type)


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(img1, img2):
    
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()


def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def bgr2ycbcr(img, only_y=True):
    '''bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def ycbcr2rgb(img):
    '''same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                          [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def init_random_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def calculate_cost(model, input_size=(1, 3, 224, 224)):
    input_ = torch.randn(input_size).cuda()
    macs, params = profile(model, inputs=(input_, ))
    macs, params = clever_format([macs, params], "%.3f")
    print("MACs:" + macs + ", Params:" + params)

      
def tensor2yuv(tensor_img):
    img_y = 0.299 * tensor_img[:, 0:1, :, :] + 0.587*tensor_img[:, 1:2, :, :] + 0.114*tensor_img[:, 2:3, :, :]
    img_u = 0.492 * (tensor_img[:, 1:2, :, :] - img_y)
    img_v = 0.877 * (tensor_img[:, 0:1, :, :] - img_y)
    img_yuv = torch.cat([img_y, img_u, img_v], dim=1)

    return img_yuv


def yuv2tensor(img_yuv):
    img_r = img_yuv[:, 0:1, :, :] + 1.14*img_yuv[:, 2:3, :, :]
    img_g = img_yuv[:, 0:1, :, :] -0.39*img_yuv[:,1:2,:,:]-0.58*img_yuv[:,2:3,:,:]
    img_b = img_yuv[:,0:1,:,:]+2.03*img_yuv[:,1:2,:,:]
    img_rgb = torch.cat([img_r,img_g,img_b], dim=1)
    return img_rgb


def match_colors_ds(im_ref, im_q, im_test):
    """ Estimates a color transformation matrix between im_ref and im_q. Applies the estimated transformation to
        im_test
    """
    im_ref_ds = F.interpolate(im_ref, scale_factor=1/2, mode='bilinear')
    im_q_ds = F.interpolate(im_q, scale_factor=1/2, mode='bilinear')

    # print('im_ref =', im_ref_ds.shape, 'im_q =', im_q_ds.shape, 'im_test =', im_test.shape)
    # print(torch.max(im_ref),torch.max(im_q))
    # plt.imshow(im_ref_ds[0].permute(1,2,0).cpu())
    # plt.show()
    # plt.imshow(im_q_ds[0].permute(1,2,0).cpu())
    # plt.show()


    im_ref_mean_re = im_ref_ds.view(*im_ref_ds.shape[:2], -1)
    im_q_mean_re = im_q_ds.view(*im_q_ds.shape[:2], -1)
    # print('im_ref_mean_re =',im_ref_mean_re.shape,'im_q_mean_re =',im_q_mean_re.shape)

    # Estimate color transformation matrix by minimizing the least squares error
    c_mat_all = []
    for ir, iq in zip(im_ref_mean_re, im_q_mean_re):
        # print('ir =', ir.t().shape, 'iq =', iq.t().shape)
        # print('ir =', ir.requires_grad, 'iq =', iq.requires_grad)
        c = torch.linalg.lstsq(ir.t(), iq.t())#what is the solution of least-square
        # print('c =',c.solution.requires_grad)
        # print('c_solution =',c.solution.shape)
        c = c.solution[:3]#r g b ???
        c_mat_all.append(c)

    # print('c_mat_all =',len(c_mat_all))
    c_mat = torch.stack(c_mat_all, dim=0)[0]
    # print('c_mat =',c_mat)
    im_q_mean_conv = torch.matmul(im_q_mean_re.permute(0, 2, 1), c_mat).permute(0, 2, 1)
    # print('im_q_mean_conv =',im_q_mean_conv.shape)
    im_q_mean_conv = im_q_mean_conv.view(im_q_ds.shape)
    # print('im_q_mean_conv =',im_q_mean_conv.shape)
    # plt.imshow(im_ref[0].permute(1,2,0).cpu())
    # plt.show()

    err = ((im_q_mean_conv - im_ref_ds) * 255.0).norm(dim=1)
    # print('err =',err.shape)

    # thresh = 20

    # # If error is larger than a threshold, ignore these pixels
    # valid = err < thresh
    # print('valid =',valid.shape,torch.max(valid))

    # pad = (im_q.shape[-1] - valid.shape[-1]) // 2
    # pad = [pad, pad, pad, pad]
    # valid = F.pad(valid, pad)
    # print('valid =',valid.shape,torch.max(valid))

    # upsample_factor = im_test.shape[-1] / valid.shape[-1]
    # valid = F.interpolate(valid.unsqueeze(1).float(), scale_factor=upsample_factor, mode='bilinear')
    # valid = valid > 0.9
    # print('valid =',valid.shape,torch.max(valid))

    # Apply the transformation to test image
    im_test_re = im_test.view(*im_test.shape[:2], -1)
    im_t_conv = torch.matmul(im_test_re.permute(0, 2, 1), c_mat).permute(0, 2, 1)
    im_t_conv = im_t_conv.view(im_test.shape)
    # print('im_t_conv =',im_t_conv.shape)
    # plt.imshow(im_ref[0].permute(1,2,0).cpu())
    # plt.show()

    return im_t_conv#, valid