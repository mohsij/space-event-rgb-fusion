# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os

import numpy as np
import torch

from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images_sensor_fusion_rgbonly, save_debug_images, save_debug_images_sensor_fusion, save_debug_images_sensor_fusion_overlaymethod
from utils.vis import save_batch_heatmaps
from kornia.geometry.conversions import axis_angle_to_rotation_matrix

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

logger = logging.getLogger(__name__)

def train_posehrnetfusion_nofusion(config, 
                        train_loader_event_rgb, 
                        model_event, model_rgb, model_hrnet_encoder_event, model_hrnet_encoder_rgb,
                        heatmap_loss_event, heatmap_loss_rgb,
                        optimizer_event, optimizer_rgb,
                        epoch, output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_event = AverageMeter()
    acc_event = AverageMeter()
    losses_rgb = AverageMeter()
    acc_rgb = AverageMeter()

    print("TRAINING NO FUSION")
    # switch to train mode

    end = time.time()
    for i, (input_tensor_event, target_event, target_weight_event, trans_hm_to_img_event, trans_crop_to_img_event, input_tensor_rgb, target_rgb, target_weight_rgb, trans_hm_to_img_rgb, trans_crop_to_img_rgb, meta) in enumerate(train_loader_event_rgb):
        # measure data loading time
        data_time.update(time.time() - end)

        model_hrnet_encoder_event.train()
        model_event.train()
        # get the features
        features_event = model_hrnet_encoder_event(input_tensor_event)

        # compute output from features
        outputs_event = model_event(features_event)
        
        target_event = target_event.cuda(non_blocking=True)
        target_weight_event = target_weight_event.cuda(non_blocking=True)

        output_event = outputs_event
        
        loss_event = heatmap_loss_event(output_event, target_event, target_weight_event)
        
        optimizer_event.zero_grad()
        loss_event.backward()
        optimizer_event.step()
        
        model_hrnet_encoder_event.eval()
        model_event.eval()
        
        # measure accuracy and record loss
        losses_event.update(loss_event.item(), input_tensor_event.size(0))

        _, avg_acc, cnt, pred_event = accuracy(output_event.detach().cpu().numpy(),
                                               target_event.detach().cpu().numpy())
        acc_event.update(avg_acc, cnt)
        
        model_hrnet_encoder_rgb.train()
        model_rgb.train()
        
        features_rgb = model_hrnet_encoder_rgb(input_tensor_rgb)
        outputs_rgb = model_rgb(features_rgb)

        target_rgb = target_rgb.cuda(non_blocking=True)
        target_weight_rgb = target_weight_rgb.cuda(non_blocking=True)

        output_rgb = outputs_rgb

        loss_rgb = heatmap_loss_rgb(output_rgb, target_rgb, target_weight_rgb)
        
        # # target weight doesnt matter. It's just ones
        # loss_fusion_event = heatmap_loss_fusion1(output_event, output_rgb.clone().detach(), target_weight_event)
        # loss_fusion_rgb = heatmap_loss_fusion2(output_rgb, output_event.clone().detach(), target_weight_rgb)
        
        # compute gradient and do update step
        
        optimizer_rgb.zero_grad()
        loss_rgb.backward()
        optimizer_rgb.step()
        
        model_hrnet_encoder_rgb.eval()
        model_rgb.eval()

        # measure accuracy and record loss
        losses_rgb.update(loss_rgb.item(), input_tensor_rgb.size(0))

        _, avg_acc, cnt, pred_rgb = accuracy(output_rgb.detach().cpu().numpy(),
                                             target_rgb.detach().cpu().numpy())
        acc_rgb.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss_event {loss_event.val:.7f} ({loss_event.avg:.7f})\t' \
                  'Loss_rgb {loss_rgb.val:.7f} ({loss_rgb.avg:.7f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader_event_rgb), batch_time=batch_time,
                      speed=input_tensor_rgb.size(0)/batch_time.val,
                      data_time=data_time, loss_event=losses_event, loss_rgb=losses_rgb, acc=acc_event)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss_event', losses_event.val, global_steps)
            writer.add_scalar('train_acc_event', acc_event.val, global_steps)
            writer.add_scalar('train_loss_rgb', losses_rgb.val, global_steps)
            writer.add_scalar('train_acc_rgb', acc_rgb.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}_{}'.format(os.path.join(output_dir, 'train'), epoch, i)
            save_debug_images_sensor_fusion(
                config, 
                input_tensor_event, input_tensor_rgb, 
                meta, 
                pred_event*(config.MODEL.IMAGE_SIZE[0]/float(config.MODEL.HEATMAP_SIZE[0])), pred_rgb*(config.MODEL.IMAGE_SIZE[0]/float(config.MODEL.HEATMAP_SIZE[0])), prefix)

def train_posehrnetfusion_nofusion_rgbonly(config, 
                        train_loader_event_rgb,
                        model_rgb, model_hrnet_encoder_rgb,
                        heatmap_loss_rgb,optimizer_rgb,
                        epoch, output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_rgb = AverageMeter()
    acc_rgb = AverageMeter()

    print("TRAINING NO FUSION RGB ONLY")
    # switch to train mode
    model_hrnet_encoder_rgb.train()
    model_rgb.train()
    
    end = time.time()
    for i, (input_tensor_rgb, target_rgb, target_weight_rgb, trans_hm_to_img_rgb, trans_crop_to_img_rgb, meta) in enumerate(train_loader_event_rgb):
        # measure data loading time
        data_time.update(time.time() - end)
        
        features_rgb = model_hrnet_encoder_rgb(input_tensor_rgb)
        outputs_rgb = model_rgb(features_rgb)

        target_rgb = target_rgb.cuda(non_blocking=True)
        target_weight_rgb = target_weight_rgb.cuda(non_blocking=True)

        output_rgb = outputs_rgb

        loss_rgb = heatmap_loss_rgb(output_rgb, target_rgb, target_weight_rgb)
        
        # # target weight doesnt matter. It's just ones
        # loss_fusion_event = heatmap_loss_fusion1(output_event, output_rgb.clone().detach(), target_weight_event)
        # loss_fusion_rgb = heatmap_loss_fusion2(output_rgb, output_event.clone().detach(), target_weight_rgb)
        
        # compute gradient and do update step
        
        optimizer_rgb.zero_grad()
        loss_rgb.backward()
        optimizer_rgb.step()

        # measure accuracy and record loss
        losses_rgb.update(loss_rgb.item(), input_tensor_rgb.size(0))

        _, avg_acc, cnt, pred_rgb = accuracy(output_rgb.detach().cpu().numpy(),
                                             target_rgb.detach().cpu().numpy())
        acc_rgb.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss_rgb {loss_rgb.val:.7f} ({loss_rgb.avg:.7f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader_event_rgb), batch_time=batch_time,
                      speed=input_tensor_rgb.size(0)/batch_time.val,
                      data_time=data_time, loss_rgb=losses_rgb, acc=acc_rgb)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss_rgb', losses_rgb.val, global_steps)
            writer.add_scalar('train_acc_rgb', acc_rgb.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}_{}'.format(os.path.join(output_dir, 'train'), epoch, i)
            save_debug_images_sensor_fusion_rgbonly(
                config, 
                input_tensor_rgb, 
                meta,
                target_rgb, output_rgb,
                pred_rgb*(config.MODEL.IMAGE_SIZE[0]/float(config.MODEL.HEATMAP_SIZE[0])), prefix)

def validate_posehrnetfusion_nofusion_rgbonly(
    config, 
    val_loader, 
    val_dataset, 
    model_rgb, model_hrnet_encoder_rgb, 
    heatmap_loss_rgb, 
    output_dir, tb_log_dir, writer_dict=None, epoch=0):
    
    batch_time = AverageMeter()
    losses_rgb = AverageMeter()
    acc_rgb = AverageMeter()

    # switch to evaluate mode
    model_rgb.eval()
    model_hrnet_encoder_rgb.eval()

    num_samples = len(val_dataset)
    
    keypoint_predictions_rgb = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    # 4x4 poses
    pose_predictions_rgb = np.tile(np.eye(4, dtype=np.float32), (num_samples, 1, 1))
    pose_gt_rgb = np.tile(np.eye(4, dtype=np.float32), (num_samples, 1, 1))
    filenames_rgb = []

    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input_tensor_rgb, target_rgb, target_weight_rgb, trans_hm_to_img_rgb, trans_crop_to_img_rgb, meta) in enumerate(val_loader):
            # compute output
            input_tensor_rgb = input_tensor_rgb.cuda()
            features_rgb = model_hrnet_encoder_rgb(input_tensor_rgb)
            # evaluate rgb outputs
            
            output_rgb = model_rgb(features_rgb)

            target_rgb = target_rgb.cuda(non_blocking=True)
            target_weight_rgb = target_weight_rgb.cuda(non_blocking=True)

            loss = heatmap_loss_rgb(output_rgb, target_rgb, target_weight_rgb)

            num_images = input_tensor_rgb.size(0)
            # measure accuracy and record loss
            losses_rgb.update(loss.item(), num_images)
            _, avg_acc, cnt, pred_rgb = accuracy(output_rgb.detach().cpu().numpy(),
                                             target_rgb.detach().cpu().numpy())

            acc_rgb.update(avg_acc, cnt)

            centre_rgb = meta['center_rgb'].numpy()
            scale_rgb = meta['scale_rgb'].numpy()

            preds_rgb, maxvals = get_final_preds(
                config, output_rgb.clone().cpu().numpy(), centre_rgb, scale_rgb)

            keypoint_predictions_rgb[idx:idx + num_images, :, 0:2] = preds_rgb[:, :, 0:2]
            keypoint_predictions_rgb[idx:idx + num_images, :, 2:3] = maxvals
            
            filenames_rgb.extend(meta['image_filename_rgb'])
            
            # get the rgb poses and add to the all poses predictions
            keypoints_bpnp_rgb, poses_bpnp_rgb, _ = val_dataset.pose_estimator_rgb.predict(output_rgb.clone(), trans_hm_to_img_rgb)
            poses_bpnp_rgb = poses_bpnp_rgb.clone()
            pose_predictions_rgb[idx:idx + num_images, 0:3, 0:3] = axis_angle_to_rotation_matrix(poses_bpnp_rgb[:, 0:3]).cpu().numpy()
            poses_bpnp_rgb = poses_bpnp_rgb.cpu().numpy()
            pose_predictions_rgb[idx:idx + num_images, 0:3, -1] = poses_bpnp_rgb[:, 3:].reshape((-1, 3))
            
            pose_gt_rgb[idx:idx + num_images, :, :] = meta["pose_rgb"]
            
            idx += num_images
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 6 == 0:
                msg = 'Testrgb: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses_rgb, acc=acc_rgb)
                logger.info(msg)

                prefix = '{}_{}_{}'.format(
                    os.path.join(output_dir, 'validation'), epoch, i
                )
                save_debug_images_sensor_fusion_rgbonly(
                    config, 
                    input_tensor_rgb, meta,
                    target_rgb, output_rgb,
                    pred_rgb*(config.MODEL.IMAGE_SIZE[0]/float(config.MODEL.HEATMAP_SIZE[0])), 
                    prefix)
                

        perf_indicator_rgb = val_dataset.evaluate(
            config, 
            output_dir,
            pose_gt_rgb,
            keypoint_predictions_rgb, 
            pose_predictions_rgb,
            filenames_rgb
        )

        # model_name = config.MODEL.NAME
        # if isinstance(name_values, list):
        #     for name_value in name_values:
        #         _print_name_value(name_value, model_name)
        # else:
        #     _print_name_value(name_values, model_name)

    return perf_indicator_rgb

def validate_posehrnetfusion_nofusion(
    config, 
    val_loader, 
    val_dataset, 
    model_event, model_rgb, model_hrnet_encoder_event, model_hrnet_encoder_rgb, 
    heatmap_loss_event, heatmap_loss_rgb, 
    output_dir, tb_log_dir, writer_dict=None, epoch=0):
    
    batch_time = AverageMeter()
    losses_event = AverageMeter()
    acc_event = AverageMeter()
    losses_rgb = AverageMeter()
    acc_rgb = AverageMeter()

    # switch to evaluate mode
    model_event.eval()
    model_rgb.eval()
    model_hrnet_encoder_event.eval()
    model_hrnet_encoder_rgb.eval()

    num_samples = len(val_dataset)
    keypoint_predictions_event = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    # 4x4 poses
    pose_predictions_event = np.tile(np.eye(4, dtype=np.float32), (num_samples, 1, 1))
    pose_gt_event = np.tile(np.eye(4, dtype=np.float32), (num_samples, 1, 1))
    filenames_event = []
    
    keypoint_predictions_rgb = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    # 4x4 poses
    pose_predictions_rgb = np.tile(np.eye(4, dtype=np.float32), (num_samples, 1, 1))
    pose_gt_rgb = np.tile(np.eye(4, dtype=np.float32), (num_samples, 1, 1))
    filenames_rgb = []

    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input_tensor_event, target_event, target_weight_event, trans_hm_to_img_event, trans_crop_to_img_event, input_tensor_rgb, target_rgb, target_weight_rgb, trans_hm_to_img_rgb, trans_crop_to_img_rgb, meta) in enumerate(val_loader):
            # compute output
            input_tensor_event = input_tensor_event.cuda()
            features_event = model_hrnet_encoder_event(input_tensor_event)
            
            input_tensor_rgb = input_tensor_rgb.cuda()
            features_rgb = model_hrnet_encoder_rgb(input_tensor_rgb)
            
            output_event = model_event(features_event)

            target_event = target_event.cuda(non_blocking=True)
            target_weight_event = target_weight_event.cuda(non_blocking=True)

            loss = heatmap_loss_event(output_event, target_event, target_weight_event)

            num_images = input_tensor_event.size(0)
            # measure accuracy and record loss
            losses_event.update(loss.item(), num_images)
            _, avg_acc, cnt, pred_event = accuracy(output_event.detach().cpu().numpy(),
                                             target_event.detach().cpu().numpy())

            acc_event.update(avg_acc, cnt)
            
            centre_event = meta['center_event'].numpy()
            scale_event = meta['scale_event'].numpy()

            preds_event, maxvals = get_final_preds(
                config, output_event.clone().cpu().numpy(), centre_event, scale_event)

            keypoint_predictions_event[idx:idx + num_images, :, 0:2] = preds_event[:, :, 0:2]
            keypoint_predictions_event[idx:idx + num_images, :, 2:3] = maxvals
            
            filenames_event.extend(meta['image_filename_event'])
            
            # get the event poses and add to the all poses predictions
            keypoints_bpnp_event, poses_bpnp_event, _ = val_dataset.pose_estimator_event.predict(output_event.clone(), trans_hm_to_img_event)
            poses_bpnp_event = poses_bpnp_event.clone()
            pose_predictions_event[idx:idx + num_images, 0:3, 0:3] = axis_angle_to_rotation_matrix(poses_bpnp_event[:, 0:3]).cpu().numpy()
            poses_bpnp_event = poses_bpnp_event.cpu().numpy()
            pose_predictions_event[idx:idx + num_images, 0:3, -1] = poses_bpnp_event[:, 3:].reshape((-1, 3))
            
            pose_gt_event[idx:idx + num_images, :, :] = meta["pose_event"]
            
            # evaluate rgb outputs
            
            output_rgb = model_rgb(features_rgb)

            target_rgb = target_rgb.cuda(non_blocking=True)
            target_weight_rgb = target_weight_rgb.cuda(non_blocking=True)

            loss = heatmap_loss_rgb(output_rgb, target_rgb, target_weight_rgb)

            num_images = input_tensor_rgb.size(0)
            # measure accuracy and record loss
            losses_rgb.update(loss.item(), num_images)
            _, avg_acc, cnt, pred_rgb = accuracy(output_rgb.detach().cpu().numpy(),
                                             target_rgb.detach().cpu().numpy())

            acc_rgb.update(avg_acc, cnt)

            centre_rgb = meta['center_rgb'].numpy()
            scale_rgb = meta['scale_rgb'].numpy()

            preds_rgb, maxvals = get_final_preds(
                config, output_rgb.clone().cpu().numpy(), centre_rgb, scale_rgb)

            keypoint_predictions_rgb[idx:idx + num_images, :, 0:2] = preds_rgb[:, :, 0:2]
            keypoint_predictions_rgb[idx:idx + num_images, :, 2:3] = maxvals
            
            filenames_rgb.extend(meta['image_filename_rgb'])
            
            # get the rgb poses and add to the all poses predictions
            keypoints_bpnp_rgb, poses_bpnp_rgb, _ = val_dataset.pose_estimator_rgb.predict(output_rgb.clone(), trans_hm_to_img_rgb)
            poses_bpnp_rgb = poses_bpnp_rgb.clone()
            pose_predictions_rgb[idx:idx + num_images, 0:3, 0:3] = axis_angle_to_rotation_matrix(poses_bpnp_rgb[:, 0:3]).cpu().numpy()
            poses_bpnp_rgb = poses_bpnp_rgb.cpu().numpy()
            pose_predictions_rgb[idx:idx + num_images, 0:3, -1] = poses_bpnp_rgb[:, 3:].reshape((-1, 3))
            
            pose_gt_rgb[idx:idx + num_images, :, :] = meta["pose_rgb"]
            
            idx += num_images
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.PRINT_FREQ == 0:
                msg = 'Testevent: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses_event, acc=acc_event)
                logger.info(msg)
                msg = 'Testrgb: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses_rgb, acc=acc_rgb)
                logger.info(msg)

                prefix = '{}_{}_{}'.format(
                    os.path.join(output_dir, 'validation'), epoch, i
                )
                save_debug_images_sensor_fusion(
                    config, 
                    input_tensor_event, 
                    input_tensor_rgb, meta, 
                    pred_event*(config.MODEL.IMAGE_SIZE[0]/float(config.MODEL.HEATMAP_SIZE[0])), 
                    pred_rgb*(config.MODEL.IMAGE_SIZE[0]/float(config.MODEL.HEATMAP_SIZE[0])), 
                    prefix)
                

        perf_indicator_event, perf_indicator_rgb = val_dataset.evaluate(
            config, 
            output_dir,
            pose_gt_event,
            keypoint_predictions_event, 
            pose_predictions_event,
            filenames_event,
            pose_gt_rgb,
            keypoint_predictions_rgb, 
            pose_predictions_rgb,
            filenames_rgb
        )

        # model_name = config.MODEL.NAME
        # if isinstance(name_values, list):
        #     for name_value in name_values:
        #         _print_name_value(name_value, model_name)
        # else:
        #     _print_name_value(name_values, model_name)

    return perf_indicator_event, perf_indicator_rgb

def enable_eval_dropout(model):
	for module in model.modules():
		if 'Dropout' in type(module).__name__:
			module.train()

def validate_posehrnetfusion_nofusion_uncertainty(
    config, 
    val_loader, 
    val_dataset, 
    model_event, model_rgb, model_hrnet_encoder_event, model_hrnet_encoder_rgb, 
    heatmap_loss_event, heatmap_loss_rgb, 
    output_dir, tb_log_dir, writer_dict=None, epoch=0):
    
    batch_time = AverageMeter()
    losses_event = AverageMeter()
    acc_event = AverageMeter()
    losses_rgb = AverageMeter()
    acc_rgb = AverageMeter()

    # switch to evaluate mode
    model_event.eval()
    model_rgb.eval()
    model_hrnet_encoder_event.eval()
    model_hrnet_encoder_rgb.eval()

    # achieving bayesian uncertainty using dropout
    enable_eval_dropout(model_hrnet_encoder_event)
    enable_eval_dropout(model_hrnet_encoder_rgb)
    
    uncertainty_K = 32

    num_samples = len(val_dataset)
    keypoint_predictions_event = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, uncertainty_K, 3),
        dtype=np.float32
    )
    keypoint_variances_event = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 2),
        dtype=np.float32
    )
    keypoint_means_event = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 2),
        dtype=np.float32
    )
    keypoint_covs_event = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 2, 2),
        dtype=np.float32
    )
    # 4x4 poses
    pose_predictions_event = np.tile(np.eye(4, dtype=np.float32), (num_samples, 1, 1))
    pose_gt_event = np.tile(np.eye(4, dtype=np.float32), (num_samples, 1, 1))
    filenames_event = []
    
    keypoint_predictions_rgb = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, uncertainty_K, 3),
        dtype=np.float32
    )
    keypoint_variances_rgb = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 2),
        dtype=np.float32
    )
    keypoint_means_rgb = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 2),
        dtype=np.float32
    )
    keypoint_covs_rgb = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 2, 2),
        dtype=np.float32
    )
    # 4x4 poses
    pose_predictions_rgb = np.tile(np.eye(4, dtype=np.float32), (num_samples, 1, 1))
    pose_gt_rgb = np.tile(np.eye(4, dtype=np.float32), (num_samples, 1, 1))
    filenames_rgb = []

    idx = 0
    
    with torch.no_grad():
        end = time.time()
        for i, (input_tensor_event, target_event, target_weight_event, trans_hm_to_img_event, trans_crop_to_img_event, input_tensor_rgb, target_rgb, target_weight_rgb, trans_hm_to_img_rgb, trans_crop_to_img_rgb, meta) in enumerate(val_loader):
            # compute output
            pred_event_list = []
            for uncertainty_iter in range(uncertainty_K):
                input_tensor_event = input_tensor_event.cuda()
                features_event = model_hrnet_encoder_event(input_tensor_event)
                
                output_event = model_event(features_event)

                target_event = target_event.cuda(non_blocking=True)
                target_weight_event = target_weight_event.cuda(non_blocking=True)

                loss = heatmap_loss_event(output_event, target_event, target_weight_event)

                num_images = input_tensor_event.size(0)
                # measure accuracy and record loss
                losses_event.update(loss.item(), num_images)
                _, avg_acc, cnt, pred_event = accuracy(output_event.detach().cpu().numpy(),
                                                target_event.detach().cpu().numpy())

                acc_event.update(avg_acc, cnt)
                
                centre_event = meta['center_event'].numpy()
                scale_event = meta['scale_event'].numpy()

                preds_event, maxvals = get_final_preds(
                    config, output_event.clone().cpu().numpy(), centre_event, scale_event)
                keypoint_predictions_event[idx:idx + num_images, :, uncertainty_iter, 0:2] = preds_event[:, :, 0:2]
                keypoint_predictions_event[idx:idx + num_images, :, uncertainty_iter, 2:3] = maxvals
                pred_event_list.append(preds_event)
                
            # get variance of pred_event_list into preds_event
            # shape of preds_event_stacked (batch, keypoints, 2, uncertainty_k)
            preds_event_stacked = np.stack(pred_event_list, axis=-1)
            preds_event_variance = np.var(preds_event_stacked, axis=-1)
            preds_event_mean = np.mean(preds_event_stacked, axis=-1)
            # calculate covariance matrices
            for batch_index in range(preds_event_stacked.shape[0]):
                for keypoint_index in range(preds_event_stacked.shape[1]):
                    cov = np.cov(preds_event_stacked[batch_index, keypoint_index, :, :])
                    keypoint_covs_event[idx:idx+batch_index, keypoint_index, :, :] = cov

            keypoint_variances_event[idx:idx + num_images, :, 0:2] = preds_event_variance[:, :, 0:2]
            keypoint_means_event[idx:idx + num_images, :, 0:2] = preds_event_mean[:, :, 0:2]
            
            filenames_event.extend(meta['image_filename_event'])
            
            # get the event poses and add to the all poses predictions
            keypoints_bpnp_event, poses_bpnp_event, _ = val_dataset.pose_estimator_event.predict(output_event.clone(), trans_hm_to_img_event)
            poses_bpnp_event = poses_bpnp_event.clone()
            pose_predictions_event[idx:idx + num_images, 0:3, 0:3] = axis_angle_to_rotation_matrix(poses_bpnp_event[:, 0:3]).cpu().numpy()
            poses_bpnp_event = poses_bpnp_event.cpu().numpy()
            pose_predictions_event[idx:idx + num_images, 0:3, -1] = poses_bpnp_event[:, 3:].reshape((-1, 3))
            
            pose_gt_event[idx:idx + num_images, :, :] = meta["pose_event"]
            
            # evaluate rgb outputs
            pred_rgb_list = []
            for uncertainty_iter in range(uncertainty_K):
                input_tensor_rgb = input_tensor_rgb.cuda()
                features_rgb = model_hrnet_encoder_rgb(input_tensor_rgb)
                
                output_rgb = model_rgb(features_rgb)

                target_rgb = target_rgb.cuda(non_blocking=True)
                target_weight_rgb = target_weight_rgb.cuda(non_blocking=True)

                loss = heatmap_loss_rgb(output_rgb, target_rgb, target_weight_rgb)

                num_images = input_tensor_rgb.size(0)
                # measure accuracy and record loss
                losses_rgb.update(loss.item(), num_images)
                _, avg_acc, cnt, pred_rgb = accuracy(output_rgb.detach().cpu().numpy(),
                                                target_rgb.detach().cpu().numpy())

                acc_rgb.update(avg_acc, cnt)

                centre_rgb = meta['center_rgb'].numpy()
                scale_rgb = meta['scale_rgb'].numpy()

                preds_rgb, maxvals = get_final_preds(
                    config, output_rgb.clone().cpu().numpy(), centre_rgb, scale_rgb)
                keypoint_predictions_rgb[idx:idx + num_images, :, uncertainty_iter, 0:2] = preds_rgb[:, :, 0:2]
                keypoint_predictions_rgb[idx:idx + num_images, :, uncertainty_iter, 2:3] = maxvals
                pred_rgb_list.append(preds_rgb)

            # get variance of pred_rgb_list into preds_rgb
            # shape of preds_rgb_stacked (batch, keypoints, 2, uncertainty_k)
            preds_rgb_stacked = np.stack(pred_rgb_list, axis=-1)
            preds_rgb_variance = np.var(preds_rgb_stacked, axis=-1)
            preds_rgb_mean = np.mean(preds_rgb_stacked, axis=-1)
            # calculate covariance matrices
            for batch_index in range(preds_rgb_stacked.shape[0]):
                for keypoint_index in range(preds_rgb_stacked.shape[1]):
                    cov = np.cov(preds_rgb_stacked[batch_index, keypoint_index, :, :])
                    keypoint_covs_rgb[idx:idx+batch_index, keypoint_index, :, :] = cov

            keypoint_variances_rgb[idx:idx + num_images, :, 0:2] = preds_rgb_variance[:, :, 0:2]
            keypoint_means_rgb[idx:idx + num_images, :, 0:2] = preds_rgb_mean[:, :, 0:2]

            
            filenames_rgb.extend(meta['image_filename_rgb'])
            
            # get the rgb poses and add to the all poses predictions
            keypoints_bpnp_rgb, poses_bpnp_rgb, _ = val_dataset.pose_estimator_rgb.predict(output_rgb.clone(), trans_hm_to_img_rgb)
            poses_bpnp_rgb = poses_bpnp_rgb.clone()
            pose_predictions_rgb[idx:idx + num_images, 0:3, 0:3] = axis_angle_to_rotation_matrix(poses_bpnp_rgb[:, 0:3]).cpu().numpy()
            poses_bpnp_rgb = poses_bpnp_rgb.cpu().numpy()
            pose_predictions_rgb[idx:idx + num_images, 0:3, -1] = poses_bpnp_rgb[:, 3:].reshape((-1, 3))
            
            pose_gt_rgb[idx:idx + num_images, :, :] = meta["pose_rgb"]
            
            idx += num_images
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.PRINT_FREQ == 0:
                msg = 'Testevent: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses_event, acc=acc_event)
                logger.info(msg)
                msg = 'Testrgb: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses_rgb, acc=acc_rgb)
                logger.info(msg)

                prefix = '{}_{}_{}'.format(
                    os.path.join(output_dir, 'validation'), epoch, i
                )
                heatmaps_dir = 'heatmaps_gaussian'
                # save_batch_heatmaps(input_tensor_event, output_event, '{}/{}_pred_hm_event.png'.format(heatmaps_dir, i))
                # save_batch_heatmaps(input_tensor_rgb, output_rgb, '{}/{}_pred_hm_rgb.png'.format(heatmaps_dir, i))
                # save_batch_heatmaps(input_tensor_event, target_event, '{}/{}_gt_hm_event.png'.format(heatmaps_dir, i))
                # save_batch_heatmaps(input_tensor_rgb, target_rgb, '{}/{}_gt_hm_rgb.png'.format(heatmaps_dir, i))
                save_debug_images_sensor_fusion(
                    config, 
                    input_tensor_event, 
                    input_tensor_rgb, meta, 
                    pred_event*(config.MODEL.IMAGE_SIZE[0]/float(config.MODEL.HEATMAP_SIZE[0])), 
                    pred_rgb*(config.MODEL.IMAGE_SIZE[0]/float(config.MODEL.HEATMAP_SIZE[0])), 
                    prefix)
                
                

        perf_indicator_event, perf_indicator_rgb = val_dataset.evaluate_with_uncertainty(
            config, 
            output_dir,
            pose_gt_event,
            keypoint_predictions_event,
            keypoint_variances_event,
            keypoint_means_event,
            keypoint_covs_event,
            pose_predictions_event,
            filenames_event,
            pose_gt_rgb,
            keypoint_predictions_rgb, 
            keypoint_variances_rgb,
            keypoint_means_rgb,
            keypoint_covs_rgb,
            pose_predictions_rgb,
            filenames_rgb
        )

        # model_name = config.MODEL.NAME
        # if isinstance(name_values, list):
        #     for name_value in name_values:
        #         _print_name_value(name_value, model_name)
        # else:
        #     _print_name_value(name_values, model_name)

    return perf_indicator_event, perf_indicator_rgb
