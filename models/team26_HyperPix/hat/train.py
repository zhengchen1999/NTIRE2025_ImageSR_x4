# flake8: noqa
import os.path as osp

import team26_DAT.archs
import team26_DAT.data
import team26_DAT.models
from models.team26_DAT.basicsr.train import train_pipeline

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
