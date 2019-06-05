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
from dataset import get_training_set, get_validation_set, get_test_set
from utils import Logger, ImbalancedDatasetSampler
from train import train_epoch
from validation import val_epoch
import test

from tensorboardX import SummaryWriter

import optuna


def objective(trial):
    opt = parse_opts()

    if trial:
        opt.weight_decay = trial.suggest_uniform('weight_decay', 0.01, 0.1)
        opt.learning_rate = trial.suggest_uniform('learning_rate', 1-5, 1-4)

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

    model, parameters = generate_model(opt)
    print(model)
    criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    # norm_method = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    if not opt.no_train:
        assert opt.train_crop in ['random', 'corner', 'center']
        if opt.train_crop == 'random':
            crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'corner':
            crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'center':
            crop_method = MultiScaleCornerCrop(
                opt.scales, opt.sample_size, crop_positions=['c'])
        spatial_transform = Compose([
            crop_method,
            RandomHorizontalFlip(),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = TemporalRandomCrop(opt.sample_duration)
        target_transform = ClassLabel()
        training_data = get_training_set(opt, spatial_transform,
                                         temporal_transform, target_transform)
        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt.batch_size,
            # sampler option is mutually exclusive with shuffle
            shuffle=False,
            sampler=ImbalancedDatasetSampler(training_data),
            num_workers=opt.n_threads,
            pin_memory=True)
        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'loss', 'acc', 'lr'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])

        optimizer = optim.Adam(
            parameters, lr=opt.learning_rate, weight_decay=opt.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, verbose=True, factor=0.1 ** 0.5)
    if not opt.no_val:
        spatial_transform = Compose([
            Scale(opt.sample_size),
            CenterCrop(opt.sample_size),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = LoopPadding(opt.sample_duration)
        target_transform = ClassLabel()
        validation_data = get_validation_set(
            opt, spatial_transform, temporal_transform, target_transform)
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=opt.batch_size,
            shuffle=False,
            sampler=ImbalancedDatasetSampler(validation_data),
            num_workers=opt.n_threads,
            pin_memory=True)
        val_logger = Logger(
            os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss', 'acc'])

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])

    print('run')
    writer = SummaryWriter(
        comment=f"_wd{opt.weight_decay}_lr{opt.learning_rate}_ft_begin{opt.ft_begin_index}_pretrain{not opt.pretrain_path == ''}")
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            epoch, losses_avg, accuracies_avg = train_epoch(i, train_loader, model, criterion, optimizer, opt,
                                                            train_logger, train_batch_logger)
            writer.add_scalar('loss/train', losses_avg, epoch)
            writer.add_scalar('acc/train', accuracies_avg, epoch)

        if not opt.no_val:
            epoch, val_losses_avg, val_accuracies_avg = val_epoch(i, val_loader, model, criterion, opt,
                                                                  val_logger)
            writer.add_scalar('loss/val', val_losses_avg, epoch)
            writer.add_scalar('acc/val', val_accuracies_avg, epoch)

        if not opt.no_train and not opt.no_val:
            scheduler.step(val_losses_avg)
        print('=' * 100)

    if opt.test:
        spatial_transform = Compose([
            Scale(int(opt.sample_size / opt.scale_in_test)),
            CornerCrop(opt.sample_size, opt.crop_position_in_test),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = LoopPadding(opt.sample_duration)
        target_transform = VideoID()

        test_data = get_test_set(opt, spatial_transform, temporal_transform,
                                 target_transform)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        test.test(test_loader, model, opt, test_data.class_names)

    writer.close()
    return val_losses_avg


def main():
    opt = parse_opts()
    print('=' * 100)
    print(f'OPTUNA_TRIALS = {opt.optuna_trials}')
    print('=' * 100)
    if opt.optuna_trials:
        study = optuna.create_study()
        study.optimize(objective, n_trials=opt.optuna_trials)
        print(study.best_params)
    else:
        objective(None)


if __name__ == '__main__':
    main()
