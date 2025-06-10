import os
import torch
import cv2
import numpy as np
import glob
from .model import MDRCT

def main(model_dir, input_path, output_path, device):
    # Configuration (match your inference.py parameters)
    window_size = 16
    scale = 4
    tile = None       # Set default tile size
    tile_overlap = 32 # Default overlap

    # Initialize model
    model = MDRCT(upscale=scale, in_chans=3, img_size=64, window_size=window_size,
                compress_ratio=3, squeeze_factor=30, conv_scale=0.01,
                overlap_ratio=0.5, img_range=1., depths=[6]*12,
                embed_dim=180, num_heads=[6]*12, gc=32, mlp_ratio=2,
                upsampler='pixelshuffle', resi_connection='1conv')

    # Load weights
    model_path = os.path.join(model_dir, "team06_MDRCT.pth")
    model.load_state_dict(torch.load(model_path)['params'], strict=True)
    model = model.to(device)
    model.eval()

    # Process images
    os.makedirs(output_path, exist_ok=True)
    
    for path in sorted(glob.glob(os.path.join(input_path, '*'))):
        imgname = os.path.splitext(os.path.basename(path))[0]
        print(f'Processing: {imgname}')
        
        try:
            # Load and preprocess image
            img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
            img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
            img = img.unsqueeze(0).to(device)
            
            # Pad input to be divisible by window_size
            _, _, h_old, w_old = img.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h_old + h_pad, :]
            img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w_old + w_pad]

            # Inference with tiling
            with torch.no_grad():
                output = process_image(img, model, window_size, scale, tile, tile_overlap)
                output = output[..., :h_old * scale, :w_old * scale]

            # Save result
            output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            cv2.imwrite(os.path.join(output_path, f'{imgname}.png'), 
                       (output * 255.0).round().astype(np.uint8))
            
        except Exception as e:
            print(f'Error processing {imgname}: {str(e)}')

def process_image(img_lq, model, window_size, scale, tile, tile_overlap):
    """ Unified processing function with tiling support """
    if tile is None:
        return model(img_lq)
    else:
        return tiled_inference(img_lq, model, window_size, scale, tile, tile_overlap)

def tiled_inference(img_lq, model, window_size, scale, tile, tile_overlap):
    """ Tile-based processing from your original inference.py """
    b, c, h, w = img_lq.size()
    tile = min(tile, h, w)
    stride = tile - tile_overlap
    
    h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
    w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
    
    E = torch.zeros(b, c, h*scale, w*scale).type_as(img_lq)
    W = torch.zeros_like(E)

    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
            out_patch = model(in_patch)
            
            E[..., h_idx*scale:(h_idx+tile)*scale, 
                w_idx*scale:(w_idx+tile)*scale].add_(out_patch)
            W[..., h_idx*scale:(h_idx+tile)*scale, 
                w_idx*scale:(w_idx+tile)*scale].add_(torch.ones_like(out_patch))

    return E.div_(W)
