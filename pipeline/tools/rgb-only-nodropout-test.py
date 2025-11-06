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
from core.loss import JointsMSELoss
from core.function import validate_posehrnetfusion_nofusion_rgbonly
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary

from utils.transforms import EventNormalise, FlipBlackEventsToWhite, FillEventBlack, RandomEventNoise, RandomEventPatchNoise

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
        cfg, cfg.DATASET.ROOT, 'test')

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
        cfg, is_train=True, dropout_prob=0.1
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

    model_rgb = torch.nn.DataParallel(hrnet_rgb, device_ids=cfg.GPUS).cuda()
    model_hrnet_encoder_rgb = torch.nn.DataParallel(hrnet_encoder_rgb, device_ids=cfg.GPUS).cuda()

    heatmap_loss_rgb = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    # Data loading code
    normalize_rgb = transforms.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]
    )
    
    valid_dataset_event_rgb = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT_ADVERSARIAL, cfg.DATASET.TEST_SET, False, cfg.DATASET.FRAMES_DIR_ADVERSARIAL,
        transforms_rgb=transforms.Compose([
            transforms.ToTensor(),
            # normalize_rgb,
        ]),
    )

    valid_loader_event_rgb = torch.utils.data.DataLoader(
        valid_dataset_event_rgb,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    checkpoint_file = cfg.TEST.MODEL_FILE

    checkpoint_file_rgb = checkpoint_file.replace("checkpoint", "checkpoint_rgb")

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file_rgb):
        logger.info("=> loading rgb checkpoint '{}'".format(checkpoint_file_rgb))
        checkpoint = torch.load(checkpoint_file_rgb)
        model_rgb.load_state_dict(checkpoint['state_dict_rgb'])
        model_hrnet_encoder_rgb.load_state_dict(checkpoint['state_dict_encoder_rgb'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file_rgb, checkpoint['epoch']))

    # evaluate on validation set
    perf_indicator_rgb = validate_posehrnetfusion_nofusion_rgbonly(
        cfg, valid_loader_event_rgb, valid_dataset_event_rgb,
        model_rgb, model_hrnet_encoder_rgb,
        heatmap_loss_rgb,
        final_output_dir, tb_log_dir, writer_dict=writer_dict
    )       

    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
