import cv2
import glob
import numpy as np
import os
import torch
from models.team21_SAKSRNet.model import SRResBlock as net
torch.cuda.empty_cache()

def main(model_dir, input_path, output_path, device=None):

    folder = 'testset'
    folder = input_path
    save_dir = output_path
    window_size = 8
    os.makedirs(save_dir, exist_ok=True)    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_init(window_size)
    model.eval()
    model = model.to(device)

    for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
        (imgname, imgext) = os.path.splitext(os.path.basename(path))
        print(f'{path=}')
        img_lq = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)

        # inference
        with torch.no_grad():
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            output = model(img_lq)
            output = output[..., :h_old * 4, :w_old * 4]

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)  
        cv2.imwrite(f'{save_dir}/{imgname}.png', output)
    
def model_init(window_size):
    model = net(upscale=4, in_chans=3, img_size=64, window_size=window_size,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
    param_key_g = 'params'

    pretrained_model = torch.load('model_zoo/team21_SAKSRNet.pth')
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)

    return model

if __name__ == '__main__':
    main()