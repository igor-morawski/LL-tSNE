"""
Copy to mmdetection/tools/
"""
import argparse
import os
import os.path as osp

import torch
from mmdet.apis import init_detector, async_inference_detector
from mmdet.utils.contextmanagers import concurrent

from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.parallel import MMDataParallel

from mmdet.apis import single_gpu_test

import cv2
import numpy as np
import mmcv
import os
import os.path as op
import tqdm


from mmcv import Config, DictAction

from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector, build_backbone, build_neck

# async def main():
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Use mmdet backbone to extract features')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--output_dir', type=str, default="features")
    parser.add_argument('--dataset_dir', type=str)
    args = parser.parse_args()
    assert op.exists(args.dataset_dir)
    config_file = args.config
    checkpoint_file = args.checkpoint
    device = 'cuda:0'

    if not op.exists(args.output_dir):
        os.mkdir(args.output_dir)
    output_filepath = op.join(args.output_dir, op.split(
        args.checkpoint)[-1].split(".")[-2]+".pkl")

    cfg = Config.fromfile(args.config)
    # build the model and load checkpoint, and the backbone
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg).backbone #only backbone

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=1,
        dist=False,
        shuffle=False)

    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    
    classes = checkpoint['meta']['CLASSES']

    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    print(classes)