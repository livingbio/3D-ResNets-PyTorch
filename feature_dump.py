import os
import sys
import json
import math
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model
from mean import get_mean, get_std
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from temporal_transforms import LoopPadding, TemporalRandomCrop
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from dataset import get_validation_set
from utils import Logger, ImbalancedDatasetSampler

from tensorboardX import SummaryWriter


def main():
    opt = parse_opts()

    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
    print(opt)
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)

    model, _ = generate_model(opt)

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    spatial_transform = Compose([
        Scale(int(opt.sample_size / opt.scale_in_test)),
        CornerCrop(opt.sample_size, opt.crop_position_in_test),
        ToTensor(opt.norm_value), norm_method
    ])
    temporal_transform = LoopPadding(opt.sample_duration)
    target_transform = VideoID()

    validation_data = get_validation_set(opt, spatial_transform, temporal_transform,
                                         target_transform)
    val_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=True)
    dump(val_loader, model, opt, validation_data.class_names)


"""
================================
Feature dump
================================

Based on test.py

Predict 512 dim feature via pretrained model
Then dump to JSON file
"""


def dump(data_loader, model, opt, class_names):
    print('Feature dump')

    model.eval()

    segment_to_feature = {}
    segment_index = 0
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(data_loader):
            outputs = model(inputs)
            assert outputs.shape[1] == 512
            for j in range(outputs.size(0)):
                label_video = targets[j]
                print(
                    f'Dump feature for {label_video}, segment_index: {segment_index}')
                label, path = label_video.split('/')
                folder, file_name = path.split('__')
                feature = outputs[j].tolist()
                segment_to_feature[f'{label_video}_{segment_index}'] = {
                    'folder': folder,
                    'file_name': file_name,
                    'label': label,
                    'feature': feature,
                    'feature_dim': len(feature),
                    'segment_index': segment_index
                }
                segment_index += 1

        with open(
                os.path.join(opt.result_path, 'segment_to_feature_34.json'),
                'w') as f:
            json.dump(segment_to_feature, f)


if __name__ == '__main__':
    main()
