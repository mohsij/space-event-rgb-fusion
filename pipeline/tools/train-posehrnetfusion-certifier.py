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
from core.function_certifier import train_posehrnetfusion_certifier
from core.function_certifier import validate_posehrnetfusion_certifier
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary

from utils.transforms import EventNormalise, FillEventBlack, RandomEventNoise, RandomEventPatchNoise, FlipBlackEventsToWhite, FillEventBlack

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
    
    hrnet_event = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True
    )

    hrnet_encoder_event = eval('models.'+cfg.MODEL.NAME+'.get_encoder')(
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

    model_event = torch.nn.DataParallel(hrnet_event, device_ids=cfg.GPUS).cuda()
    model_hrnet_encoder_event = torch.nn.DataParallel(hrnet_encoder_event, device_ids=cfg.GPUS).cuda() 

    # define loss function (criterion) and optimizer
    heatmap_loss_event = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    # Data loading code
    normalize_event = transforms.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]
    )
    
    train_dataset_event_real = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT_ADVERSARIAL, cfg.DATASET.TRAIN_SET_ADVERSARIAL, False, cfg.DATASET.FRAMES_DIR_ADVERSARIAL,
        transforms_event=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.ToTensor(),
            FlipBlackEventsToWhite(),
            FillEventBlack(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ]),
    )
    
    valid_dataset_event = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT_ADVERSARIAL, cfg.DATASET.TRAIN_SET_ADVERSARIAL, False, cfg.DATASET.FRAMES_DIR_ADVERSARIAL,
        transforms_event=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.ToTensor(),
            FlipBlackEventsToWhite(),
            FillEventBlack(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            # normalize_event,
        ]),
    )
    
    train_loader_event_real = torch.utils.data.DataLoader(
        train_dataset_event_real,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU_ADVERSARIAL_SET*len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    
    valid_loader_event = torch.utils.data.DataLoader(
        valid_dataset_event,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    best_perf_event = 1000.0
    best_model_event = False
    last_epoch = -1

    optimizer_event = get_optimizer(cfg, model_event)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH

    checkpoint_file_event = os.path.join(
        final_output_dir, 'checkpoint_certifier.pth'
    )
    
    # load the pretrained hrnet model
    pretrained = cfg.MODEL.PRETRAINED
    logger.info("=> loading checkpoint '{}'".format(pretrained))
    checkpoint = torch.load(pretrained)
    model_event.load_state_dict(checkpoint['state_dict_event'])
    model_hrnet_encoder_event.load_state_dict(checkpoint['state_dict_encoder_event'])

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


    lr_scheduler_event = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_event, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )

    eps = 100.0

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        lr_scheduler_event.step()

        # train for one epoch
        train_posehrnetfusion_certifier(
            cfg, train_loader_event_real, train_dataset_event_real,
            model_event, model_hrnet_encoder_event,
            heatmap_loss_event,
            optimizer_event,
            epoch,
            eps,
            final_output_dir, tb_log_dir, writer_dict)


        # evaluate on validation set
        perf_indicator_event = validate_posehrnetfusion_certifier(
            cfg,
            valid_loader_event, valid_dataset_event,
            model_event, model_hrnet_encoder_event,
            heatmap_loss_event,
            final_output_dir, tb_log_dir, writer_dict=writer_dict, epoch=epoch
        )
        
        eps = eps * 0.975
            
        if perf_indicator_event <= best_perf_event:
            best_perf_event = perf_indicator_event
            best_model_event = True
        else:
            best_model_event = True
            
        if best_model_event:
            logger.info('=> saving event checkpoint to {}'.format(final_output_dir))
            save_checkpoint({
                'epoch': epoch + 1,
                'model': cfg.MODEL.NAME,
                'state_dict_event': model_event.state_dict(),
                'state_dict_encoder_event': model_hrnet_encoder_event.state_dict(),
                'perf_event': perf_indicator_event,
                'optimizer_event': optimizer_event.state_dict(),
            }, best_model_event, final_output_dir, filename='checkpoint_certifier.pth')
    
    final_model_state_file = os.path.join(
        final_output_dir, 'final_state_event.pth'
    )
    logger.info('=> saving final event model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model_event.module.state_dict(), final_model_state_file)
    
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
