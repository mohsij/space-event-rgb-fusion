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
from core.function import train_posehrnetfusion_nofusion
from core.function import validate_posehrnetfusion_nofusion
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary

from utils.transforms import RandomBars, EventNormalise, FlipBlackEventsToWhite, FillEventBlack, RandomEventNoise, RandomEventPatchNoise, RandomUniformNoise, RandomColourNoise, RandomBloomV2, RandomBlur

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
    torch.multiprocessing.set_start_method('spawn')
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    hrnet_event = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True
    )
    
    hrnet_rgb = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True
    )
    
    hrnet_encoder_event = eval('models.'+cfg.MODEL.NAME+'.get_encoder')(
        cfg, is_train=True
    )

    hrnet_encoder_rgb = eval('models.'+cfg.MODEL.NAME+'.get_encoder')(
        cfg, is_train=True
    )

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

    model_event = torch.nn.DataParallel(hrnet_event, device_ids=cfg.GPUS).cuda()
    model_rgb = torch.nn.DataParallel(hrnet_rgb, device_ids=cfg.GPUS).cuda()
    model_hrnet_encoder_event = torch.nn.DataParallel(hrnet_encoder_event, device_ids=cfg.GPUS).cuda()
    model_hrnet_encoder_rgb = torch.nn.DataParallel(hrnet_encoder_rgb, device_ids=cfg.GPUS).cuda()

    # define loss function (criterion) and optimizer
    heatmap_loss_event = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    heatmap_loss_rgb = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()
    
    # heatmap_loss_fusion1 = JointsMSELoss(
    #     use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    # ).cuda()
    
    # heatmap_loss_fusion2 = JointsMSELoss(
    #     use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    # ).cuda()

    # Data loading code
    normalize_rgb = transforms.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]
    )
    train_dataset_event_rgb = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True, cfg.DATASET.FRAMES_DIR,
        transforms_event=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.ToTensor(),
            FlipBlackEventsToWhite(),
            FillEventBlack(),
            transforms.RandomApply([RandomEventNoise(brighten=False)], p=0.5),
            transforms.RandomApply([RandomEventNoise(brighten=True)], p=0.5),
            transforms.RandomApply([RandomEventPatchNoise(brighten=False)], p=0.5),
            transforms.RandomApply([RandomEventPatchNoise(brighten=True)], p=0.5),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.RandomApply([RandomBlur()], p=0.25),
            transforms.RandomApply([RandomBars()], p=0.5),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25]),
        ]),
        transforms_rgb=transforms.Compose([
            transforms.ToPILImage(),
            #transforms.RandomApply([RandomBloomV2()], p=0.2),
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.5, contrast=0.75, hue=0.2),
            transforms.RandomApply([RandomUniformNoise()], p=0.5),
            transforms.RandomApply([RandomColourNoise()], p=0.5),
            transforms.RandomApply([RandomBlur()], p=0.25),
            transforms.RandomApply([RandomBars(0.0)], p=0.5),
            # normalize_rgb,
        ]),
    )
    
    valid_dataset_event_rgb = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT_ADVERSARIAL, cfg.DATASET.TEST_SET, False, cfg.DATASET.FRAMES_DIR_ADVERSARIAL,
        transforms_event=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.ToTensor(),
            FlipBlackEventsToWhite(),
            FillEventBlack(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5]),
        ]),
        transforms_rgb=transforms.Compose([
            transforms.ToTensor(),
            # normalize_rgb,
        ]),
    )

    train_loader_event_rgb = torch.utils.data.DataLoader(
        train_dataset_event_rgb,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    
    valid_loader_event_rgb = torch.utils.data.DataLoader(
        valid_dataset_event_rgb,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    best_perf_event = 1000.0
    best_model_event = False
    best_perf_rgb = 1000.0
    best_model_rgb = False
    last_epoch = -1
    optimizer_event = get_optimizer(cfg, model_event)
    optimizer_rgb = get_optimizer(cfg, model_rgb)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file_event = os.path.join(
        final_output_dir, 'checkpoint_event.pth'
    )
    checkpoint_file_rgb = os.path.join(
        final_output_dir, 'checkpoint_rgb.pth'
    )

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file_event):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file_event))
        checkpoint = torch.load(checkpoint_file_event)
        begin_epoch = checkpoint['epoch']
        best_perf_event = checkpoint['perf_event']
        last_epoch = checkpoint['epoch']
        model_event.load_state_dict(checkpoint['state_dict_event'])
        model_hrnet_encoder_event.load_state_dict(checkpoint['state_dict_encoder_event'])
        optimizer_event.load_state_dict(checkpoint['optimizer_event'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file_event, checkpoint['epoch']))

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file_rgb):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file_rgb))
        checkpoint = torch.load(checkpoint_file_rgb)
        begin_epoch = checkpoint['epoch']
        best_perf_rgb = checkpoint['perf_rgb']
        last_epoch = checkpoint['epoch']
        model_rgb.load_state_dict(checkpoint['state_dict_rgb'])
        model_hrnet_encoder_rgb.load_state_dict(checkpoint['state_dict_encoder_rgb'])
        optimizer_rgb.load_state_dict(checkpoint['optimizer_rgb'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file_rgb, checkpoint['epoch']))


    lr_scheduler_event = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_event, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )
    
    lr_scheduler_rgb = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_rgb, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        lr_scheduler_event.step()
        lr_scheduler_rgb.step()

        # train for one epoch
        train_posehrnetfusion_nofusion(cfg, train_loader_event_rgb, model_event, model_rgb, model_hrnet_encoder_event, model_hrnet_encoder_rgb, 
            heatmap_loss_event, heatmap_loss_rgb,
            optimizer_event, optimizer_rgb, epoch,
            final_output_dir, tb_log_dir, writer_dict)


        # evaluate on validation set
        perf_indicator_event, perf_indicator_rgb = validate_posehrnetfusion_nofusion(
            cfg, valid_loader_event_rgb, valid_dataset_event_rgb,
            model_event, model_rgb, 
            model_hrnet_encoder_event, model_hrnet_encoder_rgb,
            heatmap_loss_event, heatmap_loss_rgb,
            final_output_dir, tb_log_dir, writer_dict=writer_dict, epoch=epoch
        )

        if perf_indicator_event <= best_perf_event:
            best_perf_event = perf_indicator_event
            best_model_event = True
        else:
            best_model_event = True
            
        if perf_indicator_rgb <= best_perf_rgb:
            best_perf_rgb = perf_indicator_rgb
            best_model_rgb = True
        else:
            best_model_rgb = True


        if best_model_event:
            logger.info('=> saving event checkpoint to {}'.format(final_output_dir))
            save_checkpoint({
                'epoch': epoch + 1,
                'model': cfg.MODEL.NAME,
                'state_dict_event': model_event.state_dict(),
                'state_dict_encoder_event': model_hrnet_encoder_event.state_dict(),
                'perf_event': perf_indicator_event,
                'optimizer_event': optimizer_event.state_dict(),
            }, best_model_event, final_output_dir, filename='checkpoint_event.pth')
            
        if best_model_rgb:
            logger.info('=> saving rgb checkpoint to {}'.format(final_output_dir))
            save_checkpoint({
                'epoch': epoch + 1,
                'model': cfg.MODEL.NAME,
                'state_dict_rgb': model_rgb.state_dict(),
                'state_dict_encoder_rgb': model_hrnet_encoder_rgb.state_dict(),
                'perf_rgb': perf_indicator_rgb,
                'optimizer_rgb': optimizer_rgb.state_dict(),
            }, best_model_rgb, final_output_dir, filename='checkpoint_rgb.pth')

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state_event.pth'
    )
    logger.info('=> saving final event model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model_event.module.state_dict(), final_model_state_file)
    
    final_model_state_file = os.path.join(
        final_output_dir, 'final_state_rgb.pth'
    )
    logger.info('=> saving final rgb model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model_rgb.module.state_dict(), final_model_state_file)
    
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
