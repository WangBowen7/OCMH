import numpy as np

import clip
import torch
import torchvision

import h5py
from PIL import Image
import hdf5storage

from transformers import set_seed

from utils import get_logger

import pandas as pd
import os
import argparse
import yaml
import importlib

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default='MIRFlickr', help='Dataset name: MIRFlickr/MSCOCO/NUSWIDE')
    parser.add_argument('--bits', type=int, default=16, help='16/32/64/128')
    parser.add_argument('--model', type=str, default='cpah', help='model name: djsrh/jdsh/dgcpn/cirh/ucch/dcmh/cpah')

    parser.add_argument('--LR_ORIGIN', type=float, default=0.01, help='The epoch of training stage.') 
    parser.add_argument('--LR_EXPAND', type=float, default=10, help='The epoch of training stage.') 
    parser.add_argument('--LR_IMG', type=float, default=0.001, help='The epoch of training stage.')
    parser.add_argument('--LR_TXT', type=float, default=0.001, help='The epoch of training stage.')
    
    parser.add_argument('--LAMBDA0', type=float, default=1, help='The epoch of training stage.')
    parser.add_argument('--LAMBDA1', type=float, default=1, help='The epoch of training stage.')
    parser.add_argument('--LAMBDA2', type=float, default=0.1, help='The epoch of training stage.')
    
    parser.add_argument('--split_ratio', type=int, default=37, help='The epoch of training stage.')
    parser.add_argument('--log', type=str, default='main_lr_0.05_0419', help='log path')
    parser.add_argument('--log_best', type=str, default='best_main_lr_0.05_0419', help='best log path')
    parser.add_argument('--save_log', type=bool, default=False, help='best log path')
    parser.add_argument('--mode', type=str, default='lr_new', help='best log path') 
    parser.add_argument('--set_alpha', type=int, default=1, help='best log path') 
    
    args = parser.parse_args()    
    yaml_path = './config/config.yaml'
    assert os.path.isfile(yaml_path), "cfg file: {} not found".format(yaml_path)
    
    # merge config
    with open(yaml_path, 'r') as yaml_file:
        yaml_config = yaml.safe_load(yaml_file)
    args_dict = vars(args)
    args_dict.update(yaml_config)
    args = argparse.Namespace(**args_dict)

        
    dataset_args = getattr(args, args.dataname)
    args_dict = vars(args)
    if dataset_args:
        args_dict.update(dataset_args)
    args = argparse.Namespace(**args_dict)
    
    return args

def main(args):
    set_seed(seed=2023)
    logger, stream_log, txt_log = get_logger()
    trainer = importlib.import_module('.{}_trainer_lr'.format('ours'.lower()), package='train')
    trainer = getattr(trainer, '{}_Trainer'.format('OURS'.upper()))(args, logger)
    trainer.train()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    
    args = parse_args()
    
    main(args)







