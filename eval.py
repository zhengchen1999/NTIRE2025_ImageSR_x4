import os
import torch
from PIL import Image
from tqdm import tqdm
import torch.multiprocessing as mp
from torchvision import transforms
import torchvision.transforms.functional as F
import csv
import pyiqa
import cv2
import numpy as np
from einops import rearrange
import csv

from utils import utils_image as util

def read_csv_to_dict(filename):
    data = {}

    with open(filename, mode='r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)

        for row in csv_reader:
            key = row[csv_reader.fieldnames[0]]
            data[key] = {
                field: (float(value) if is_number(value) else value)
                for field, value in row.items() if field != csv_reader.fieldnames[0]
            }

    return data


def is_number(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def rgb_to_ycrcb(tensor):
    tensor_np = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    ycrcb_np = cv2.cvtColor(tensor_np, cv2.COLOR_RGB2YCrCb)
    ycrcb_tensor = torch.tensor(ycrcb_np).permute(2, 0, 1).unsqueeze(0).float()
    return ycrcb_tensor


class IQA:
    def __init__(self, device=None):
        self.device = device if device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.iqa_metrics = {
            'lpips': pyiqa.create_metric('lpips', device=self.device),
            'dists': pyiqa.create_metric('dists', device=self.device),
            'niqe': pyiqa.create_metric('niqe', device=self.device),
            'musiq': pyiqa.create_metric('musiq', device=self.device),
            'maniqa': pyiqa.create_metric('maniqa', device=self.device),
            'clipiqa': pyiqa.create_metric('clipiqa', device=self.device)
        }

    def calculate_values(self, output_image, target_image):
        if target_image is not None:
            assert type(output_image) == type(target_image), "The types of output_image and target_image do not match"

        if type(output_image) == torch.Tensor or type(output_image) == np.ndarray:
            if type(output_image) == np.ndarray:
                output_image = torch.tensor(output_image).contiguous().float()
                if target_image is not None:
                    target_image = torch.tensor(target_image).contiguous().float()

            if len(output_image.shape) == 3:

                output_image = output_image.unsqueeze(0)
                if target_image is not None:
                    target_image = target_image.unsqueeze(0)

            if output_image.shape[-1] == 3:
                print("Rearranging image dimensions from (N, W, H, C) to (N, C, W, H)")
                output_image = rearrange(output_image, "b h w c -> b c h w").contiguous().float()
                if target_image is not None:
                    target_image = rearrange(target_image, "b h w c -> b c h w").contiguous().float()
            elif output_image.shape[-1] == 4:
                output_image = output_image[:, :, :, :3]
                print("Rearranging image dimensions from (N, W, H, C) to (N, C, W, H)")
                output_image = rearrange(output_image, "b h w c -> b c h w").contiguous().float()
                if target_image is not None:
                    target_image = rearrange(target_image, "b h w c -> b c h w").contiguous().float()

            output_tensor = output_image.to(self.device)
            if target_image is not None:
                target_tensor = target_image.to(self.device)
            else:
                target_tensor = None
        else:
            output_tensor = F.to_tensor(output_image).unsqueeze(0).to(self.device)
            if target_image is not None:
                target_tensor = F.to_tensor(target_image).unsqueeze(0).to(self.device)
            else:
                target_tensor = None

        if target_tensor is not None and output_tensor.shape != target_tensor.shape:
            print(f"[IQA Reshape] predicted shape: {output_tensor.shape}, target shape: {target_tensor.shape}")
            min_height = min(output_tensor.shape[2], target_tensor.shape[2])
            min_width = min(output_tensor.shape[3], target_tensor.shape[3])
            resize_transform = transforms.Resize((min_height, min_width))
            output_tensor = resize_transform(output_tensor)
            target_tensor = resize_transform(target_tensor)

        try:
            if target_tensor is not None:
                lpips_value = self.iqa_metrics['lpips'](output_tensor, target_tensor)
                dists_value = self.iqa_metrics['dists'](output_tensor, target_tensor)


            niqe_value = self.iqa_metrics['niqe'](output_tensor)
            musiq_value = self.iqa_metrics['musiq'](output_tensor)
            maniqa_value = self.iqa_metrics['maniqa'](output_tensor)
            clipiqa_value = self.iqa_metrics['clipiqa'](output_tensor)

            result = {}

            if target_tensor is not None:
                result['LPIPS'] = lpips_value.item()
                result['DISTS'] = dists_value.item()

            result['NIQE'] = niqe_value.item()
            result['MUSIQ'] = musiq_value.item()
            result['MANIQA'] = maniqa_value.item()
            result['CLIP-IQA'] = clipiqa_value.item()
        except Exception as e:
            print(f"Error: {e}")
            return None

        return result


def calculate_iqa_for_partition(output_folder, target_folder, output_files, device, rank):
    iqa = IQA(device=device)
    local_results = {}
    for output_file in tqdm(output_files, total=len(output_files), desc=f"Processing images on GPU {rank}"):
        if target_folder is not None:
            target_file = output_file.replace('x4', '')

        output_image_path = os.path.join(output_folder, output_file)
        output_image = Image.open(output_image_path)

        if target_folder is not None:
            target_image_path = os.path.join(target_folder, target_file)
            assert os.path.exists(target_image_path), f"No such path: {target_image_path}"

            target_image = Image.open(target_image_path)
        else:
            target_image = None

        values = iqa.calculate_values(output_image, target_image)
        values["psnr"], values["ssim"] = util.cal_psnr_ssim(output_image_path, target_image_path)
        if values is not None:
            local_results[output_file] = values

    return local_results


def main_worker(rank, gpu_id, output_folder, target_folder, output_files, return_dict, num_gpus):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    partition_size = len(output_files) // num_gpus
    start_idx = rank * partition_size
    end_idx = (rank + 1) * partition_size if rank != num_gpus - 1 else len(output_files)

    output_files_partition = output_files[start_idx:end_idx]

    local_results = calculate_iqa_for_partition(output_folder, target_folder, output_files_partition,
                                                device, rank)
    return_dict[rank] = local_results


import argparse

if __name__ == "__main__":
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=str, default="output_dir")
    parser.add_argument("--target_folder", type=str, default="div2k-val/HR")
    parser.add_argument("--metrics_save_path", type=str, default="./IQA_results")
    parser.add_argument("--gpu_ids", type=str, default="0")
    args = parser.parse_args()

    resume_figs = []
    output_files = sorted([f for f in os.listdir(args.output_folder) if f.endswith('.png') and f not in resume_figs])
    if args.target_folder is not None:
        target_files = sorted(
            [f for f in os.listdir(args.target_folder) if f.endswith('.png') and f not in resume_figs])

        assert len(output_files) == len(target_files), \
            (f"The number of output images should be equal to the number of target images: "
             f"{len(output_files)} != {len(target_files)}")
    else:
        target_files = None

    manager = mp.Manager()
    return_dict = manager.dict()

    args.gpu_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(',')]
    print(f"Using GPU: {args.gpu_ids}")
    num_gpus = len(args.gpu_ids)

    processes = []
    for rank, gpu_id in enumerate(args.gpu_ids):
        p = mp.Process(target=main_worker, args=(
        rank, gpu_id, args.output_folder, args.target_folder, output_files, return_dict, num_gpus))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    results = {}
    for rank in return_dict.keys():
        results.update(return_dict[rank])

    folder_name = os.path.basename(args.output_folder)
    parent_folder = os.path.dirname(args.output_folder)
    next_level_folder = os.path.basename(parent_folder)
    os.makedirs(args.metrics_save_path, exist_ok=True)
    average_results_filename = f"{args.metrics_save_path}/{next_level_folder}--{folder_name}.txt"
    results_filename = f"{args.metrics_save_path}/{next_level_folder}--{folder_name}.csv"

    if results:
        all_keys = set()
        for values in results.values():
            try:
                all_keys.update(values.keys())
            except Exception as e:
                print(f"Error: {e}")

        all_keys = sorted(all_keys)

        average_results = {}
        for key in all_keys:
            average_results[key] = np.mean([values.get(key, 0) for values in results.values()])

        with open(average_results_filename, 'w') as f:
            for key, value in average_results.items():
                f.write(f"{key}: {value}\n")

        print("Average:")
        print(average_results)

        with open(results_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Filename'] + list(all_keys))
            for filename, values in results.items():
                row = [filename] + [values.get(key, '') for key in all_keys]
                writer.writerow(row)

        print(f"IQA results have been saved to {results_filename} file")

