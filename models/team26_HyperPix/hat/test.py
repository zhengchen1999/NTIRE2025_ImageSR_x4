# flake8: noqa
import os.path as osp

import team26_DAT.hat.archs
import team26_DAT.hat.data
import team26_DAT.hat.models
from models.team26_DAT.basicsr.test import test_pipeline

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
