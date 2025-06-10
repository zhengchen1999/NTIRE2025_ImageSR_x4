import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
from utils.common import *

class L1Loss(nn.Module):

    def __init__(self):
        super(L1Loss, self).__init__()
        self.criterion = torch.nn.L1Loss()

    def forward(self, x, y):
        return self.criterion(x,y)
class FFTLoss(nn.Module):
    def __init__(self):
        super(FFTLoss, self).__init__()
        self.L1 = L1Loss()

    def forward(self, x, y):
        pred_fft = torch.rfft(x, signal_ndim=2, normalized=False, onesided=False)
        gt_fft = torch.rfft(y, signal_ndim=2, normalized=False, onesided=False)

        loss = self.L1(pred_fft, gt_fft)
        return loss
    
class multi_VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, p_resize=True, lam_p=1.0, lam_l=0.5, lam_c=1):
        super(multi_VGGPerceptualLoss, self).__init__()
        self.loss_fn = VGGPerceptualLoss(p_resize)
        self.lam_p = lam_p
        self.lam_l = lam_l
        self.color_loss = ColorLoss(lam_c)

    def forward(self, out3, out2, out1, gt1, feature_layers=[2], mask=None):
        gt2 = F.interpolate(gt1, scale_factor=0.5, mode='bilinear', align_corners=False)
        gt3 = F.interpolate(gt1, scale_factor=0.25, mode='bilinear', align_corners=False)
        if mask is not None:
            mask2 = F.interpolate(mask, scale_factor=0.5, mode='bilinear', align_corners=False)
            mask3 = F.interpolate(mask, scale_factor=0.25, mode='bilinear', align_corners=False)
            loss1 = self.lam_p*self.loss_fn(out1, gt1, feature_layers=feature_layers, mask=mask) + self.lam_l*F.l1_loss(out1*mask, gt1*mask) + self.color_loss(out1*mask, gt1*mask)
            loss2 = self.lam_p*self.loss_fn(out2, gt2, feature_layers=feature_layers, mask=mask2) + self.lam_l*F.l1_loss(out2*mask2, gt2*mask2) + self.color_loss(out2*mask, gt2*mask)
            loss3 = self.lam_p*self.loss_fn(out3, gt3, feature_layers=feature_layers, mask=mask3) + self.lam_l*F.l1_loss(out3*mask3, gt3*mask3) + self.color_loss(out3*mask, gt3*mask)
        else: 
            loss1 = self.lam_p*self.loss_fn(out1, gt1, feature_layers=feature_layers) + self.lam_l*F.l1_loss(out1, gt1) + self.color_loss(out1, gt1)
            loss2 = self.lam_p*self.loss_fn(out2, gt2, feature_layers=feature_layers) + self.lam_l*F.l1_loss(out2, gt2) + self.color_loss(out2, gt2)
            loss3 = self.lam_p*self.loss_fn(out3, gt3, feature_layers=feature_layers) + self.lam_l*F.l1_loss(out3, gt3) + self.color_loss(out3, gt3)
        return loss1+loss2+loss3
        
class FocalFrequencyLoss(nn.Module):
    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False, batch_matrix=False):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x):
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, (
            'Patch factor should be divisible by image height and width')
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])

        y = torch.stack(patch_list, 1)

        if torch.__version__.split('+')[0].split('.') > ['1', '7', '1']:
            freq = torch.fft.fft2(y, norm='ortho')
            freq = torch.stack([freq.real, freq.imag], -1)
        else:
            freq = torch.rfft(y, 2, onesided=False, normalized=True)
        return freq

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        if matrix is not None:
            weight_matrix = matrix.detach()
        else:
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            weight_matrix = torch.clamp(matrix_tmp, min=0.0, max=1.0).clone().detach()

        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]
        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    def forward(self, pred, target, matrix=None):
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight


class CombinedLoss(nn.Module):
    def __init__(self, p_resize=True, lam_p=1.0, lam_l=0.5, lam_c=1):
        """
        Combined loss that includes multi-scale VGG perceptual, color, and focal frequency losses.

        Args:
            p_resize (bool): Whether to resize in VGG perceptual loss.
            lam_p (float): Weight for perceptual loss.
            lam_l (float): Weight for L1 loss.
            lam_c (float): Weight for color loss.
        """
        super(CombinedLoss, self).__init__()
        
        # Initialize the multi-scale VGG perceptual loss and color loss
        self.multi_vgg_loss = multi_VGGPerceptualLoss(p_resize, lam_p, lam_l, lam_c)
        
        # Initialize focal frequency loss with default parameters
        self.focal_freq_loss = FocalFrequencyLoss(loss_weight=1.0, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False, batch_matrix=False)
        
        # Default weight for focal frequency loss; this can be adjusted manually
        self.lam_f = 1.0  

    def forward(self, out3, out2, out1, gt1, feature_layers=[2], mask=None):
        """
        Forward method to calculate combined loss.

        Args:
            out3, out2, out1 (torch.Tensor): Outputs at different scales.
            gt1 (torch.Tensor): Ground truth image.
            feature_layers (list): Layers used in perceptual loss.
            mask (torch.Tensor, optional): Mask for selective loss calculation.
        """
        # Compute the multi-scale VGG perceptual, color, and L1 losses
        vgg_perceptual_loss = self.multi_vgg_loss(out3, out2, out1, gt1, feature_layers=feature_layers, mask=mask)
        
        # Calculate focal frequency loss between the highest scale output and ground truth
        frequency_loss = self.focal_freq_loss(out1, gt1)
        
        # Combine losses with specified weights
        total_loss = vgg_perceptual_loss + self.lam_f * frequency_loss
        return total_loss
    
    
class L1VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, lam_p=1.0, lam_l=0.5):
        super(L1VGGPerceptualLoss, self).__init__()
        self.lam_p = lam_p
        self.lam_l = lam_l
        self.loss_fn = VGGPerceptualLoss()

    def forward(self, out, gt, feature_layers=[2], mask=None):
        if mask is not None:
            loss = self.lam_p*self.loss_fn(out, gt, feature_layers=feature_layers, mask=mask) + self.lam_l*F.l1_loss(out*mask, gt*mask)
        else:
            loss = self.lam_p*self.loss_fn(out, gt, feature_layers=feature_layers) + self.lam_l*F.l1_loss(out, gt)
        return loss
        


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.resize = resize

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[], mask=None, return_feature=False):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
            
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                if mask is not None:
                    _,_,H,W = x.shape
                    mask_resized = F.interpolate(mask, size=(H, W), mode='nearest')[:, 0:1, :, :]
                    x = x*mask_resized
                    y = y*mask_resized
                    loss += torch.nn.functional.l1_loss(x, y)
                else:
                    loss += torch.nn.functional.l1_loss(x, y)
                    
                if return_feature:
                    return x, y
                    
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss
    
import torch.nn as nn
from basicsr.utils.registry import LOSS_REGISTRY
import utils.SWT as SWT
import pywt
import numpy as np

@LOSS_REGISTRY.register()
class SWTLoss(nn.Module):
    def __init__(self, args, loss_weight_ll=0.05, loss_weight_lh=0.01, loss_weight_hl=0.01, loss_weight_hh=0.00, reduction='mean'):
        super(SWTLoss, self).__init__()
        self.loss_weight_ll = loss_weight_ll
        self.loss_weight_lh = loss_weight_lh
        self.loss_weight_hl = loss_weight_hl
        self.loss_weight_hh = loss_weight_hh
        self.WAVELET = args.WAVELET
        self.criterion = nn.L1Loss(reduction=reduction)
        self.multi_vgg_loss = multi_VGGPerceptualLoss(lam_p=args.WEIGHT_PEC, lam_l=args.WEIGHT_L1).to("cuda")
        # Initialize the wavelet transform filters
        wavelet = pywt.Wavelet('sym19')
        dlo = wavelet.dec_lo
        an_lo = np.divide(dlo, sum(dlo))
        an_hi = wavelet.dec_hi
        rlo = wavelet.rec_lo
        syn_lo = 2 * np.divide(rlo, sum(rlo))
        syn_hi = wavelet.rec_hi
        filters = pywt.Wavelet('wavelet_normalized', [an_lo, an_hi, syn_lo, syn_hi])
        self.sfm = SWT.SWTForward(1, filters, 'periodic').to("cuda")  # Assume using CUDA

    def forward(self, out3, out2, out1, gt1, feature_layers=[2], mask=None):
        gt2 = F.interpolate(gt1, scale_factor=0.5, mode='bilinear', align_corners=False)
        gt3 = F.interpolate(gt1, scale_factor=0.25, mode='bilinear', align_corners=False)
        vgg_perceptual_loss = self.multi_vgg_loss(out3, out2, out1, gt1, feature_layers=feature_layers, mask=mask)
        if mask is not None:
            mask2 = F.interpolate(mask, scale_factor=0.5, mode='bilinear', align_corners=False)
            mask3 = F.interpolate(mask, scale_factor=0.25, mode='bilinear', align_corners=False)
            loss1 = self.compute_wavelet_loss(out1 * mask, gt1 * mask)
            loss2 = self.compute_wavelet_loss(out2 * mask2, gt2 * mask2)
            loss3 = self.compute_wavelet_loss(out3 * mask3, gt3 * mask3)
        else:
            loss1 = self.compute_wavelet_loss(out1, gt1)
            loss2 = self.compute_wavelet_loss(out2, gt2)
            loss3 = self.compute_wavelet_loss(out3, gt3)
        return self.WAVELET * (loss1 + loss2 + loss3) + vgg_perceptual_loss

    def compute_wavelet_loss(self, pred, target):
        # Convert images to Y channel
        sr_img_y = 16.0 + (pred[:, 0:1, :, :] * 65.481 + pred[:, 1:2, :, :] * 128.553 + pred[:, 2:, :, :] * 24.966)
        hr_img_y = 16.0 + (target[:, 0:1, :, :] * 65.481 + target[:, 1:2, :, :] * 128.553 + target[:, 2:, :, :] * 24.966)
        
        # Compute wavelet transforms
        wavelet_sr = self.sfm(sr_img_y)[0]
        wavelet_hr = self.sfm(hr_img_y)[0]
        
        # Extract subbands
        LL_sr, LH_sr, HL_sr, HH_sr = wavelet_sr[:, 0:1, :, :], wavelet_sr[:, 1:2, :, :], wavelet_sr[:, 2:3, :, :], wavelet_sr[:, 3:, :, :]
        LL_hr, LH_hr, HL_hr, HH_hr = wavelet_hr[:, 0:1, :, :], wavelet_hr[:, 1:2, :, :], wavelet_hr[:, 2:3, :, :], wavelet_hr[:, 3:, :, :]
        
        # Compute losses for each subband
        loss_subband_LL = self.loss_weight_ll * self.criterion(LL_sr, LL_hr)
        loss_subband_LH = self.loss_weight_lh * self.criterion(LH_sr, LH_hr)
        loss_subband_HL = self.loss_weight_hl * self.criterion(HL_sr, HL_hr)
        loss_subband_HH = self.loss_weight_hh * self.criterion(HH_sr, HH_hr)
        
        return loss_subband_LL + loss_subband_LH + loss_subband_HL + loss_subband_HH
        
def color_loss(x1, x2):
	x1_norm = torch.sqrt(torch.sum(torch.pow(x1, 2)) + 1e-8)
	x2_norm = torch.sqrt(torch.sum(torch.pow(x2, 2)) + 1e-8)
	# 内积
	x1_x2 = torch.sum(torch.mul(x1, x2))
	cosin = 1 - x1_x2 / (x1_norm * x2_norm)
	return cosin
        
class ColorLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0):
        super(ColorLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        if isinstance(pred,tuple):
            loss_1 = color_loss(pred[0], target)
            target_2 = F.interpolate(target, scale_factor=0.5, mode='bilinear')
            loss_2 = color_loss(pred[1], target_2)
            target_3 = F.interpolate(target_2, scale_factor=0.5, mode='bilinear')
            loss_3 = color_loss(pred[2], target_3)
            return self.loss_weight * (loss_1+loss_2+loss_3)
        else:
            return self.loss_weight * color_loss(pred, target)
        
        
