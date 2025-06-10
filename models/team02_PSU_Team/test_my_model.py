import os
import cv2
import glob
import numpy as np
from tqdm import tqdm





import os
import torch
import torch.nn as nn 
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from geomloss import SamplesLoss  # Geometric Losses library for Sinkhorn divergence
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
##############################
# REPLACE THIS BLOCK WITH:
# from models.ssva_net_schrodinger_bridge_advanced import SSVA_Net
# IF YOU PREFER IMPORTING
# THE MODEL FROM YOUR FILE.
##############################
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0,1,3,2,4,5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / ((H // window_size) * (W // window_size)))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0,1,3,2,4,5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size -1)*(2*window_size -1), num_heads)
        )

        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1,2,0).contiguous()
        relative_coords[:,:,0] += self.window_size -1
        relative_coords[:,:,1] += self.window_size -1
        relative_coords[:,:,0] *= 2 * self.window_size -1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size*self.window_size, self.window_size*self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2,0,1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1,2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        window_size = min(self.window_size, H, W)
        shift_size = self.shift_size if window_size > self.shift_size else 0

        pad_b = (window_size - H % window_size) % window_size
        pad_r = (window_size - W % window_size) % window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = x.shape[1], x.shape[2]

        if shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1,2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, window_size)
        x_windows = x_windows.view(-1, window_size*window_size, C)
        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.view(-1, window_size, window_size, C)
        shifted_x = window_reverse(attn_windows, window_size, H_pad, W_pad)

        if shift_size > 0:
            x = torch.roll(shifted_x, shifts=(shift_size, shift_size), dims=(1,2))
        else:
            x = shifted_x

        x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H*W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class SwinTransformerLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size=8, mlp_ratio=4.0):
        super(SwinTransformerLayer, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio
                )
            )
    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)
        return x

############################################
# Multi-Scale Refinement Network with Attention
############################################
class MultiScaleRefineNet(nn.Module):
    def __init__(self, base_channels=64):
        super(MultiScaleRefineNet, self).__init__()
        # We'll process the image at two scales: original and half
        # At the coarse scale, we refine a downsampled version and then upsample and combine.

        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Coarse scale refinement
        self.coarse_refine = self._build_refine_block(in_channels=6, base_channels=base_channels)
        # Fine scale refinement
        self.fine_refine = self._build_refine_block(in_channels=6, base_channels=base_channels)

    def _build_refine_block(self, in_channels, base_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels*2, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*4, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*4, base_channels*4, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels*2, base_channels, 2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 3, 3, 1, 1)
        )

    # def forward(self, pred, shadow):
    #     # pred: preliminary prediction
    #     # shadow: original shadow image
    #     inp_full = torch.cat([pred, shadow], dim=1)
    #     # Downsample both for coarse scale
    #     pred_down = self.down(pred)
    #     shadow_down = self.down(shadow)
    #     inp_down = torch.cat([pred_down, shadow_down], dim=1)

    #     coarse_out = self.coarse_refine(inp_down)
    #     coarse_up = self.up(coarse_out)

    #     # Combine coarse refinement as guidance
    #     refined_input = torch.cat([pred + coarse_up, shadow], dim=1)
    #     fine_out = self.fine_refine(refined_input)
    #     return fine_out
    def forward(self, pred, shadow):
        # pred: preliminary prediction
        # shadow: original shadow image
        
        # 1) Figure out how to pad so H and W are multiples of 16
        _, _, h, w = pred.shape
        pad_h = (16 - (h % 16)) % 16
        pad_w = (16 - (w % 16)) % 16
        
        pred = F.pad(pred, (0, pad_w, 0, pad_h), mode='reflect')
        shadow = F.pad(shadow, (0, pad_w, 0, pad_h), mode='reflect')

        # 2) Downsample both
        pred_down = self.down(pred)
        shadow_down = self.down(shadow)
        inp_down = torch.cat([pred_down, shadow_down], dim=1)

        # 3) Coarse refinement and upsample
        coarse_out = self.coarse_refine(inp_down)
        coarse_up = self.up(coarse_out)

        # 4) Combine coarse refinement with the (padded) pred
        combined = pred + coarse_up
        refined_input = torch.cat([combined, shadow], dim=1)

        # 5) Fine refinement
        fine_out = self.fine_refine(refined_input)

        # 6) Crop back to the original size
        fine_out = fine_out[:, :, :h, :w]
        
        return fine_out



############################################
# Conditional Diffusion Module
############################################
class UNet(nn.Module):
    def __init__(self, img_channels):
        super(UNet, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(img_channels * 2, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Conv2d(64, img_channels, kernel_size=3, padding=1)

    def forward(self, x, t, condition):
        x = torch.cat([x, condition], dim=1)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        u1 = self.up1(d3)
        u2 = self.up2(u1 + d2)
        out = self.out_conv(u2 + d1)
        return out

class ConditionalDiffusionModule(nn.Module):
    def __init__(self, img_channels, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super(ConditionalDiffusionModule, self).__init__()
        self.num_timesteps = num_timesteps
        beta_schedule = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        self.register_buffer('beta_schedule', beta_schedule)
        self.register_buffer('sqrt_beta', torch.sqrt(beta_schedule))
        self.register_buffer('sqrt_one_minus_beta', torch.sqrt(1 - beta_schedule))
        self.eps_model = UNet(img_channels)
    
    def forward(self, x, t, condition):
        noise = torch.randn_like(x)
        sqrt_beta_t = self.sqrt_beta[t].view(-1, 1, 1, 1)
        sqrt_one_minus_beta_t = self.sqrt_one_minus_beta[t].view(-1, 1, 1, 1)
        x_t = sqrt_one_minus_beta_t * x + sqrt_beta_t * noise
        eps = self.eps_model(x_t, t, condition)
        return eps

############################################
# PatchGAN Discriminator for Adversarial Loss
############################################
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64): 
        super(PatchDiscriminator, self).__init__()
        # LSGAN based PatchGAN
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels*2, 4, 2, 1),
            nn.InstanceNorm2d(base_channels*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels*2, base_channels*4, 4, 2, 1),
            nn.InstanceNorm2d(base_channels*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels*4, 1, 4, 1, 1) # output 1-channel score map
        )

    def forward(self, x):
        return self.model(x)

############################################
# Main Model
############################################
class SSVA_Net(nn.Module):
    def __init__(self, img_channel=3, width=32, middle_blk_num=1,
                 enc_blk_nums=[1, 1, 1, 1], dec_blk_nums=[1, 1, 1, 1],
                 d_state=64):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, kernel_size=3, padding=1, stride=1)
        self.ending = nn.Conv2d(width, img_channel, kernel_size=3, padding=1, stride=1)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                SwinTransformerLayer(
                    dim=chan,
                    depth=num,
                    num_heads=4,
                    window_size=8
                )
            )
            self.downs.append(nn.Conv2d(chan, chan * 2, kernel_size=2, stride=2))
            chan = chan * 2

        self.middle_blks = SwinTransformerLayer(
            dim=chan,
            depth=middle_blk_num,
            num_heads=8,
            window_size=8
        )

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, kernel_size=1, bias=False),
                nn.PixelShuffle(2)
            ))
            chan = chan // 2
            self.decoders.append(
                SwinTransformerLayer(
                    dim=chan,
                    depth=num,
                    num_heads=4,
                    window_size=8
                )
            )

        self.padder_size = 2 ** len(self.encoders)
        self.diffusion_module = ConditionalDiffusionModule(img_channel)
        self.ot_loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)

        # Multi-scale refinement stage
        self.refine_net = MultiScaleRefineNet(base_channels=64)

    def forward(self, inp, target=None, t=None):
        B, C, H, W = inp.shape
        inp_padded = self.check_image_size(inp)
        x = self.intro(inp_padded)
        x = x.permute(0, 2, 3, 1)
        encs = []

        # Encoder
        for encoder, down in zip(self.encoders, self.downs):
            H_e, W_e = x.shape[1], x.shape[2]
            x = x.view(B, -1, x.shape[-1])
            x = encoder(x, H_e, W_e)
            x = x.view(B, H_e, W_e, -1)
            encs.append(x)
            x = x.permute(0, 3, 1, 2)
            x = down(x)
            x = x.permute(0, 2, 3, 1)

        # Middle
        H_m, W_m = x.shape[1], x.shape[2]
        x = x.view(B, -1, x.shape[-1])
        x = self.middle_blks(x, H_m, W_m)
        x = x.view(B, H_m, W_m, -1)

        # Decoder
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = x.permute(0, 3, 1, 2)
            x = up(x)
            x = x.permute(0, 2, 3, 1)
            if x.shape[1] != enc_skip.shape[1] or x.shape[2] != enc_skip.shape[2]:
                x = x[:, :enc_skip.shape[1], :enc_skip.shape[2], :]
            x = x + enc_skip
            H_d, W_d = x.shape[1], x.shape[2]
            x = x.view(B, -1, x.shape[-1])
            x = decoder(x, H_d, W_d)
            x = x.view(B, H_d, W_d, -1)

        x = x.permute(0, 3, 1, 2)
        x = self.ending(x)
        x = x + inp_padded[:, :, :x.shape[2], :x.shape[3]]
        x = x[:, :, :H, :W]

        # refinement stage
        refined = self.refine_net(x, inp)

        if target is not None and t is not None:
            eps = self.diffusion_module(refined, t, inp)
            return refined, eps
        else:
            return refined

    def compute_ot_loss(self, pred, target):
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        loss = self.ot_loss(pred_flat, target_flat)
        return loss

    def check_image_size(self, x):
        _, _, h, w = x.size()
        pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x

############################################
# Loss Functions
############################################
def diffusion_loss(eps_pred, eps_true):
    return F.mse_loss(eps_pred, eps_true)

class VGGLoss(nn.Module):
    def __init__(self, device=DEVICE):
        super(VGGLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features.to(device).eval()
        self.vgg_layers = nn.Sequential(*list(vgg)[:36]).to(device)
        self.device = device
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        x_vgg = self.vgg_layers(x)
        y_vgg = self.vgg_layers(y)
        return nn.functional.l1_loss(x_vgg, y_vgg)

class FocalFrequencyLoss(nn.Module):
    def __init__(self, alpha=1.0, device=DEVICE):
        super(FocalFrequencyLoss, self).__init__()
        self.alpha = alpha
        self.device = device

    def forward(self, input, target):
        input = input.to(self.device)
        target = target.to(self.device)
        input_fft = torch.fft.fft2(input)
        target_fft = torch.fft.fft2(target)
        diff = input_fft - target_fft
        abs_diff = torch.abs(diff)
        loss = torch.pow(abs_diff, self.alpha)
        return torch.mean(loss)

class CombinedLoss(nn.Module):
    def __init__(self, lambda_vgg=0.01, lambda_ff=0.1, device=DEVICE):
        super(CombinedLoss, self).__init__()
        self.lambda_vgg = lambda_vgg
        self.lambda_ff = lambda_ff
        self.l1_loss = nn.L1Loss()
        self.vgg_loss = VGGLoss(device=device)
        self.ff_loss = FocalFrequencyLoss(device=device)

    def forward(self, input, target):
        l1 = self.l1_loss(input, target)
        vgg = self.vgg_loss(input, target)
        ff = self.ff_loss(input, target)
        return l1 + self.lambda_vgg * vgg + self.lambda_ff * ff

class VGG_comLoss(nn.Module):
    def __init__(self, lambda_vgg=0.01, lambda_ff=0.1, device=DEVICE):
        super(VGG_comLoss, self).__init__()
        self.lambda_vgg = lambda_vgg
        self.lambda_ff = lambda_ff
        self.l1_loss = nn.L1Loss()
        self.vgg_loss = VGGLoss(device=device)
        self.ff_loss = FocalFrequencyLoss(device=device)

    def forward(self, input, target):
        l1 = self.l1_loss(input, target)
        vgg = self.vgg_loss(input, target)
        # omit ff to reduce complexity here if needed, or keep it
        return l1 + self.lambda_vgg * vgg

############################################
# Training and Validation
############################################
import lpips

#########################################################
# Inference Script Starts Here
#########################################################
def inference_on_folder(
    model_weights_path,
    input_shadow_dir,
    output_dir="Submation",
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Loads the SSVA_Net model with given weights, processes all images in input_shadow_dir,
    and saves the output images to output_dir (same filename, same resolution).
    """

    # 1. Create the model
    model = SSVA_Net(
        img_channel=3,
        width=32,
        middle_blk_num=1,
        enc_blk_nums=[16, 16, 16, 16],
        dec_blk_nums=[16, 16, 16, 16],
        d_state=64
    )
    model.to(device)

    # 2. Load model weights
    if os.path.exists(model_weights_path):
        print(f"Loading weights from {model_weights_path}")
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
    else:
        raise FileNotFoundError(f"Weights file not found: {model_weights_path}")

    model.eval()

    # 3. Prepare transform (no resizing, just ToTensor)
    transform = transforms.ToTensor()

    # 4. Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 5. Inference loop
    with torch.no_grad():
        file_list = sorted(os.listdir(input_shadow_dir))
        for filename in file_list:
            if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif")):
                continue

            img_path = os.path.join(input_shadow_dir, filename)
            image = Image.open(img_path).convert("RGB")

            # Convert to tensor
            input_tensor = transform(image).unsqueeze(0).to(device)

            # Forward pass (no diffusion steps, just the forward function)
            pred = model(input_tensor)

            # Clamp results to [0,1]
            pred = torch.clamp(pred, 0, 1)

            # Convert tensor back to PIL
            pred_image_pil = transforms.ToPILImage()(pred.squeeze(0).cpu())

            # Save the result
            save_path = os.path.join(output_dir, filename)
            pred_image_pil.save(save_path)

            print(f"Saved: {save_path}")

    print("Inference completed!")


######################################
# Example usage (uncomment to run):
######################################
if __name__ == "__main__":
    # Modify these paths as needed:
    model_weights = "/data2/ntire/sr/team/team02/NTIRE2025_ImageSR_x4/model_zoo/super_resolution_wights.pth"  # Your trained weights
    val_shadow_dir = "/data2/ntire/sr/team/team02/NTIRE2025_ImageSR_x4/up_X4"  # Where your val shadow images are
    output_folder = "/data2/ntire/sr/result/team02"  # The folder to save results

    inference_on_folder(
        model_weights_path=model_weights,
        input_shadow_dir=val_shadow_dir,
        output_dir=output_folder,
        device="cuda:0"  # or "cpu" or another GPU device if you like
    )









