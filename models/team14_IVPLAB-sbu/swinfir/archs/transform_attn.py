
import torch
import torch.nn as nn
import pytorch_wavelets as pw





class ResBlock(nn.Module):
    def __init__(self, embed_dim):
        super(ResBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.LeakyReLU(negative_slope= 0.2, inplace= True),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        )

    def __call__(self, x):
        out = self.body(x)
        return out + x


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, embed_dim, reduction=1):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dim // reduction, embed_dim, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class WaveletUnit(nn.Module):
    def __init__(self, embed_dim, J = 4, wave = 'db4', mode = 'symmetric'):
        super(WaveletUnit, self).__init__()

        self.embed_dim = embed_dim
        self.J = J
        self.wave = wave
        self.mode = mode

        self.LL_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim , 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv_layer = nn.Sequential(
            nn.Conv2d(embed_dim * 3, embed_dim * 3, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        self.dwt = pw.DWTForward(J=self.J, wave=self.wave, mode=self.mode)
        self.inv_dwt = pw.DWTInverse(wave=self.wave, mode=self.mode)
        
    def forward(self, x):
        
        x = x
        cA, subbands = self.dwt(x)
        cA_conv = self.LL_conv(cA)    # [4, 60, 10, 10]
        sub_list = []

        for i, _ in enumerate(subbands):
            subband = subbands[i]                # subbannds: [4, 60, 3, 33, 33] ,  [4, 60, 3, 20, 20],  [4, 60, 3, 13, 13] 
            b, c, k, h, w = subband.shape
            subband = subband.reshape(b, -1 , h, w)  # [4, 60, 3, h, w] --> [4, 180, h, w]
            subband_conv = self.conv_layer(subband)      # [4, 180, h, w]  
            subband = subband_conv.view(b, c, k, h, w)     # [4, 180, h, w]--> [4, 60, 3, h, w]      
            sub_list.append(subband)

        out= self.inv_dwt((cA_conv, sub_list))    # [4, 60, 60, 60]
        return out
    

class WaveTransform(nn.Module):
    def __init__(self, embed_dim, wave = 'db4', mode = 'symmetric', *args, **kwars):
        super(WaveTransform, self).__init__()

        self.J = 4
        self.wave = wave
        self.mode = mode
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 3, 1, 1, 0),     # [4,  180, 60, 60]  --> [4, 60, 60, 60]
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.wave = WaveletUnit(embed_dim // 3, J = self.J, wave= self.wave, mode=self.mode)  # [4, 60, 60, 60]
        self.conv2 = nn.Conv2d(embed_dim // 3, embed_dim, 1, 1, 0)   # [4, 60, 60, 60] --> [4, 180, 60, 60]


    def forward(self, x):
        x = self.conv1(x)          # [4, 180, 60, 60] -> [4, 60, 60, 60]
        wave = self.wave(x)    # [4, 60, 60, 60]
        output = self.conv2(wave + x)   
        return output    
    

class FourierUnit(nn.Module):
    def __init__(self, embed_dim, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.conv_layer = torch.nn.Conv2d(embed_dim * 2, embed_dim * 2, 1, 1, 0)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]

        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(ffted)

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4,
                                                                       2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        return output



class FourierTransform(nn.Module):
    def __init__(self, embed_dim):
        # bn_layer not used
        super(FourierTransform, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.fu = FourierUnit(embed_dim // 2)

        self.conv2 = torch.nn.Conv2d(embed_dim // 2, embed_dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        output = self.fu(x)
        output = self.conv2(x + output)
        return output


class TransformAttention(nn.Module):
    def __init__(self, embed_dim):
        super(TransformAttention, self).__init__()

        self.wavelet = WaveTransform(embed_dim)
        self.fft = FourierTransform(embed_dim)
        self.channel_attn = CALayer(embed_dim)

        self.last_conv = nn.Conv2d(embed_dim * 2, embed_dim, 1, 1, 0)


    def forward(self, x):
        wavelet = self.wavelet(x)
        fft = self.fft(x)
        cat = torch.cat([wavelet, fft], dim= 1)
        last_conv = self.last_conv(cat)
        return self.channel_attn(last_conv)


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim):
        super(AttentionBlock , self).__init__()

        self.trans_attn = TransformAttention(embed_dim)   # [4, 180, 60, 60]
        self.resblock = ResBlock(embed_dim)  # [4, 180, 60, 60]
        self.fuse = nn.Conv2d(embed_dim * 2, embed_dim , 1, 1, 0)

    def forward(self, x):
        transform_attn = self.trans_attn(x)
        res_block = self.resblock(x)
        concat = torch.cat([transform_attn, res_block], dim=1)  # [4, 360, 60, 60]
        return self.fuse(concat)  # [4, 180, 60, 60]


