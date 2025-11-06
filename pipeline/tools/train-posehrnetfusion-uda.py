# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss, FeatureMSELoss
from core.function_uda import train_posehrnetfusion_uda
from core.function_uda import validate_posehrnetfusion_uda
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary

from utils.transforms import RandomBars, RandomUniformNoise, RandomColourNoise, RandomBlur

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, cfg.DATASET.ROOT_ADVERSARIAL, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    
    hrnet_rgb = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True
    )

    hrnet_encoder_rgb = eval('models.'+cfg.MODEL.NAME+'.get_encoder')(
        cfg, is_train=True
    )
    
    discriminator = models.discriminator.resnet34(num_classes=2, in_channels=cfg.MODEL.NUM_JOINTS)

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../../lib/models', cfg.MODEL.NAME + '.py'),
        final_output_dir)
    # logger.info(pprint.pformat(model))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    model_rgb = torch.nn.DataParallel(hrnet_rgb, device_ids=cfg.GPUS).cuda()
    model_hrnet_encoder_rgb = torch.nn.DataParallel(hrnet_encoder_rgb, device_ids=cfg.GPUS).cuda()
    model_discriminator = torch.nn.DataParallel(discriminator, device_ids=cfg.GPUS).cuda()
    

    # define loss function (criterion) and optimizer
    heatmap_loss_rgb = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()
    
    discriminator_loss = torch.nn.CrossEntropyLoss().cuda()

    # Data loading code
    normalize_rgb = transforms.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]
    )
    train_dataset_rgb = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True, cfg.DATASET.FRAMES_DIR,
        transforms_rgb=transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.5, contrast=0.75, hue=0.2),
            transforms.RandomApply([RandomUniformNoise()], p=0.5),
            transforms.RandomApply([RandomColourNoise()], p=0.5),
            transforms.RandomApply([RandomBlur()], p=0.25),
            transforms.RandomApply([RandomBars(0.0)], p=0.5),
            # normalize_rgb,
        ]),
    )
    
    train_dataset_rgb_real = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT_ADVERSARIAL, cfg.DATASET.TRAIN_SET_ADVERSARIAL, False, cfg.DATASET.FRAMES_DIR_ADVERSARIAL,
        transforms_rgb=transforms.Compose([
            transforms.ToTensor(),
            # normalize_rgb,
        ]),
    )
    
    valid_dataset_rgb_real = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT_ADVERSARIAL, cfg.DATASET.TRAIN_SET_ADVERSARIAL, False, cfg.DATASET.FRAMES_DIR_ADVERSARIAL,
        transforms_rgb=transforms.Compose([
            transforms.ToTensor(),
            # normalize_rgb,
        ]),
    )

    train_loader_rgb = torch.utils.data.DataLoader(
        train_dataset_rgb,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    
    train_loader_rgb_real = torch.utils.data.DataLoader(
        train_dataset_rgb_real,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU_ADVERSARIAL_SET*len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    
    valid_loader_rgb_real = torch.utils.data.DataLoader(
        valid_dataset_rgb_real,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    best_perf_rgb = 1000.0
    best_model_rgb = False
    last_epoch = -1

    optimizer_rgb = get_optimizer(cfg, model_rgb)
    optimizer_discriminator = get_optimizer(cfg, model_discriminator)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH

    checkpoint_file_rgb = os.path.join(
        final_output_dir, 'checkpoint_uda.pth'
    )
    
    # load the pretrained hrnet model
    # pretrained should be synthetic rgb trained with no dropout
    pretrained = cfg.MODEL.PRETRAINED
    logger.info("=> loading checkpoint '{}'".format(pretrained))
    checkpoint = torch.load(pretrained)
    model_rgb.load_state_dict(checkpoint['state_dict_rgb'])
    model_hrnet_encoder_rgb.load_state_dict(checkpoint['state_dict_encoder_rgb'])

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file_rgb):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file_rgb))
        checkpoint = torch.load(checkpoint_file_rgb)
        begin_epoch = checkpoint['epoch']
        best_perf_rgb = checkpoint['perf_rgb']
        last_epoch = checkpoint['epoch']
        model_rgb.load_state_dict(checkpoint['state_dict_rgb'])
        model_hrnet_encoder_rgb.load_state_dict(checkpoint['state_dict_encoder_rgb'])
        model_discriminator.load_state_dict(checkpoint['state_dict_discriminator'])
        optimizer_rgb.load_state_dict(checkpoint['optimizer_rgb'])
        optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file_rgb, checkpoint['epoch']))


    lr_scheduler_rgb = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_rgb, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )
    
    lr_scheduler_discriminator = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_discriminator, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        lr_scheduler_rgb.step()

        # train for one epoch
        train_posehrnetfusion_uda(
            cfg, train_loader_rgb, train_loader_rgb_real,
            model_rgb, model_hrnet_encoder_rgb, model_discriminator,
            heatmap_loss_rgb, discriminator_loss,
            optimizer_rgb, optimizer_discriminator,
            epoch,
            final_output_dir, tb_log_dir, writer_dict)


        # evaluate on validation set
        perf_indicator_rgb = validate_posehrnetfusion_uda(
            cfg,
            valid_loader_rgb_real, valid_dataset_rgb_real,
            model_rgb, model_hrnet_encoder_rgb,
            heatmap_loss_rgb,
            final_output_dir, tb_log_dir, writer_dict=writer_dict, epoch=epoch
        )
            
        if perf_indicator_rgb <= best_perf_rgb:
            best_perf_rgb = perf_indicator_rgb
            best_model_rgb = True
        else:
            best_model_rgb = True
            
        if best_model_rgb:
            logger.info('=> saving uda checkpoint to {}'.format(final_output_dir))
            save_checkpoint({
                'epoch': epoch + 1,
                'model': cfg.MODEL.NAME,
                'state_dict_rgb': model_rgb.state_dict(),
                'state_dict_encoder_rgb': model_hrnet_encoder_rgb.state_dict(),
                'state_dict_discriminator': model_discriminator.state_dict(),
                'perf_rgb': perf_indicator_rgb,
                'optimizer_rgb': optimizer_rgb.state_dict(),
                'optimizer_discriminator': optimizer_discriminator.state_dict(),
            }, best_model_rgb, final_output_dir, filename='checkpoint_uda.pth')
    
    final_model_state_file = os.path.join(
        final_output_dir, 'final_state_uda.pth'
    )
    logger.info('=> saving final rgb model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model_rgb.module.state_dict(), final_model_state_file)
    
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
