#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6+

import argparse
import yaml
import logging
import torch
_logger = logging.getLogger('train')

def _parse_args():
    """ Parse command-line arguments and load config file if provided. """
    # Parse config file argument first
    config_parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    config_parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                               help='YAML config file specifying default arguments')
    
    args_config, remaining_argv = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')  

    parser.add_argument('--model', default='mamba_vision_T', type=str, metavar='MODEL',
                         help='Name of model to train (default: "gc_vit_tiny"')
    parser.add_argument('--dataset_name', metavar='DIR',type=str, default='DIV2K', help='path to train dataset')
    
    
    # Please change this when tunning
    parser.add_argument('--epochs', type=int, default=1000, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument('--seed', type=int, default=42, metavar='S',help='random seed (default: 42)')
    parser.add_argument('--use_seed', type=bool, default=False,help='choose to use random seed, type boolean')
    parser.add_argument('--gpu_ids', default='0', type=str, metavar='NAME',help='available gpus')
    parser.add_argument('--WEIGHT_L1', default=0.7, type=float, metavar='N', help='')
    parser.add_argument('--WEIGHT_PEC', default=0.3, type=float, metavar='N', help='')
    
    # you could choose these parameters
    parser.add_argument('--weight_decay', type=float, default=0.0001, help="L2 regularzation coefficient")
    parser.add_argument('--momentum', type=float, default=0.9, help="momentum for SGD optimizer")
    parser.add_argument('--bn-momentum', type=float, default=0.9,help='BatchNorm momentum override (if not None)')
    parser.add_argument('--bn-eps', type=float, default=None,help='BatchNorm epsilon override (if not None)')
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training")
    parser.add_argument('--use_gpu', type=bool, default=True, help="Flag to use GPU if available")
    parser.add_argument('--gp', default=None, type=str, metavar='POOL', help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
    parser.add_argument('--no-prefetcher', action='store_true', default=True,help='disable fast prefetcher')
    parser.add_argument('-j', '--workers', type=int, default=8, metavar='N',help='how many training processes to use (default: 8)')
    parser.add_argument('--pin-mem', action='store_true', default=False,help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    
    # Set it directly to data(your FER2013 dataset path)
    parser.add_argument('--data_dir',type=str,default="./dataset")
    parser.add_argument('--json_path',type=str,default="./utils/train_X4.json")
    parser.add_argument('--output_dir',type=str, default="./output")
    parser.add_argument('--torchscript', dest='torchscript', action='store_true',help='torch.jit.script the full model')
    
    # parser.add_argument('--ckp_dir', type=str, default='./ckp', help="Path to checkpoint directory")
    parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                        help='Initialize model from this checkpoint (default: none)')
    parser.add_argument('--log_path', type=str, default='./log', help="Path to log file")
    
    # Set it to False When you Run this code first time
    parser.add_argument('--pretrained', action='store_true', default=False, help='Start with pretrained version of specified network (if avail)')
    parser.add_argument('--ckp_path', type=str, default= './ckp/VGG_attention_1_epoch/VGG_attention_1_epoch_30_.pth', help="relative path to pretrain ckp")
    
    # Set model parameters
    parser.add_argument('--img_size', type=int, default=224, help="image size for croping and resizing")
    parser.add_argument('--attn-drop-rate', type=float, default=0.0, metavar='PCT',
                        help='Drop of the attention, gaussian std')
    parser.add_argument('--drop-rate', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.38, metavar='PCT',
                        help='Drop path rate (default: None)')


    args = parser.parse_args(remaining_argv)
    args.dtype = str(torch.float16) 
    # If config file exists, load and override default arguments
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
            args = parser.parse_args(remaining_argv)  # Re-parse with new defaults

    # Save parsed arguments as a YAML string for logging purposes
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    
    return args,args_text

if __name__ == "__main__":
    args,args_text = _parse_args()
    print(args)  # Debugging: print parsed arguments
