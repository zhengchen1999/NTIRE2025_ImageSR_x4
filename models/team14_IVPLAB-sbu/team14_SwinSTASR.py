import os
import torch
import numpy as np
from PIL import Image
from .swinfir.archs.swinfir_arch import SwinFIR
from .swinfir.archs.transform_attn import ResBlock, CALayer, FourierTransform
from .swinfir.archs.transform_attn import TransformAttention, AttentionBlock
from .swinfir.archs.swinfir_utils import *
from .swinfir.archs.local_arch import *


def load_image(path):
    """Load and preprocess an image for SwinFIR."""
    img = Image.open(path).convert('RGB')
    
    # Convert to numpy array and normalize to [0, 1]
    img = np.array(img).astype(np.float32) / 255.0
    
    # Convert to tensor and add batch dimension
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # CHW-RGB -> BCHW-RGB
    return img

def save_image(tensor, path):
    """Save a tensor as an image."""
    tensor = tensor.squeeze().float().cpu().clamp_(0, 1).numpy()
    tensor = np.transpose(tensor[[2, 1, 0], :, :], (1, 2, 0))  # RGB->BGR for OpenCV
    tensor = (tensor * 255.0).round().astype(np.uint8)
    Image.fromarray(tensor).save(path)

def main(model_dir, input_path, output_path, device):
    """
    Main function for SwinSTASR inference.
    Args:
        model_dir: Path to the pretrained model.
        input_path: Folder containing input images.
        output_path: Folder to save restored images.
        device: Computation device ('cuda' or 'cpu').
    """

    # Initialize model with original architecture config
    model = SwinFIR(
        img_size=60,
        patch_size=1,
        in_chans=3,
        embed_dim=180,
        depths=[6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6],
        window_size=12,
        mlp_ratio=2,
        upscale=4,
        img_range=1.0,
        upsampler='pixelshuffle',
        resi_connection='transformattention'
    )

    model.load_state_dict(torch.load(model_dir, map_location=device)['params_ema'], strict=True)
    model.eval()
    model.to(device)    

    # Process images matching original workflow
    os.makedirs(output_path, exist_ok=True)
    # Process each image in the input folder
    for img_name in os.listdir(input_path):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_path, img_name)
            
            # Load and preprocess the image
            lq = load_image(img_path).to(device)
            
            # Pad input image to be a multiple of window_size
            _, _, h, w = lq.size()
            window_size = 12
            mod_pad_h = (h // window_size + 1) * window_size - h
            mod_pad_w = (w // window_size + 1) * window_size - w
            lq = torch.cat([lq, torch.flip(lq, [2])], 2)[:, :, :h + mod_pad_h, :]
            lq = torch.cat([lq, torch.flip(lq, [3])], 3)[:, :, :, :w + mod_pad_w]
            
            # Perform inference
            with torch.no_grad():
                output = model(lq)
            
            # Remove padding
            output = output[..., :h * 4, :w * 4]
            
            # Save the output image
            save_image(output, os.path.join(output_path, img_name))

    print(f"Results saved to {output_path}")