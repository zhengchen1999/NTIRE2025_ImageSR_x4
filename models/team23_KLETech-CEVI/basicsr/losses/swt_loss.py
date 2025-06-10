import torch.nn as nn
from basicsr.utils.registry import LOSS_REGISTRY
import basicsr.losses.SWT as SWT
import pywt
import numpy as np

@LOSS_REGISTRY.register()
class SWTLoss(nn.Module):
    def __init__(self, loss_weight_ll=0.01, loss_weight_lh=0.01, loss_weight_hl=0.01, loss_weight_hh=0.01, reduction='mean'):
        super(SWTLoss, self).__init__()
        self.loss_weight_ll = loss_weight_ll
        self.loss_weight_lh = loss_weight_lh
        self.loss_weight_hl = loss_weight_hl
        self.loss_weight_hh = loss_weight_hh

        self.criterion = nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):
        wavelet = pywt.Wavelet('sym19')
            
        dlo = wavelet.dec_lo
        an_lo = np.divide(dlo, sum(dlo))
        an_hi = wavelet.dec_hi
        rlo = wavelet.rec_lo
        syn_lo = 2*np.divide(rlo, sum(rlo))
        syn_hi = wavelet.rec_hi

        filters = pywt.Wavelet('wavelet_normalized', [an_lo, an_hi, syn_lo, syn_hi])
        sfm = SWT.SWTForward(1, filters, 'periodic').to("cuda")

        ## wavelet bands of sr image
        sr_img_y       = 16.0 + (pred[:,0:1,:,:]*65.481 + pred[:,1:2,:,:]*128.553 + pred[:,2:,:,:]*24.966)
        # sr_img_cb      = 128 + (-37.797 *pred[:,0:1,:,:] - 74.203 * pred[:,1:2,:,:] + 112.0* pred[:,2:,:,:])
        # sr_img_cr      = 128 + (112.0 *pred[:,0:1,:,:] - 93.786 * pred[:,1:2,:,:] - 18.214 * pred[:,2:,:,:])

        wavelet_sr  = sfm(sr_img_y)[0]

        LL_sr   = wavelet_sr[:,0:1, :, :]
        LH_sr   = wavelet_sr[:,1:2, :, :]
        HL_sr   = wavelet_sr[:,2:3, :, :]
        HH_sr   = wavelet_sr[:,3:, :, :]     

        ## wavelet bands of hr image
        hr_img_y       = 16.0 + (target[:,0:1,:,:]*65.481 + target[:,1:2,:,:]*128.553 + target[:,2:,:,:]*24.966)
        # hr_img_cb      = 128 + (-37.797 *target[:,0:1,:,:] - 74.203 * target[:,1:2,:,:] + 112.0* target[:,2:,:,:])
        # hr_img_cr      = 128 + (112.0 *target[:,0:1,:,:] - 93.786 * target[:,1:2,:,:] - 18.214 * target[:,2:,:,:])
     
        wavelet_hr     = sfm(hr_img_y)[0]

        LL_hr   = wavelet_hr[:,0:1, :, :]
        LH_hr   = wavelet_hr[:,1:2, :, :]
        HL_hr   = wavelet_hr[:,2:3, :, :]
        HH_hr   = wavelet_hr[:,3:, :, :]

        loss_subband_LL = self.loss_weight_ll * self.criterion(LL_sr, LL_hr)
        loss_subband_LH = self.loss_weight_lh * self.criterion(LH_sr, LH_hr)
        loss_subband_HL = self.loss_weight_hl * self.criterion(HL_sr, HL_hr)
        loss_subband_HH = self.loss_weight_hh * self.criterion(HH_sr, HH_hr)

        return loss_subband_LL + loss_subband_LH + loss_subband_HL + loss_subband_HH