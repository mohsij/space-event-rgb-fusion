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
from utils.vis import save_debug_images, save_debug_images_uda, save_debug_images_uda_test
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

def train_posehrnetfusion_uda(config, 
                        train_loader_rgb, train_loader_rgb_real,
                        model_rgb, model_hrnet_encoder_rgb, model_discriminator,
                        heatmap_loss_rgb, discriminator_loss,
                        optimizer_rgb, optimizer_discriminator,
                        epoch, output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_rgb = AverageMeter()
    acc_rgb = AverageMeter()

    # switch to train mode
    model_rgb.train()
    model_discriminator.train()
    
    model_hrnet_encoder_rgb.train()
    
    real_set_iter = iter(train_loader_rgb_real)

    end = time.time()
    for i, (input_tensor_rgb, target_rgb, target_weight_rgb, trans_hm_to_img_rgb, trans_crop_to_img_rgb, meta) in enumerate(train_loader_rgb):
        try:
            input_tensor_rgb_real,target_rgb_real,target_weight_rgb_real,_,_,meta_real = next(real_set_iter)
        except StopIteration:
            real_set_iter = iter(train_loader_rgb_real)
            input_tensor_rgb_real,target_rgb_real,target_weight_rgb_real,_,_,meta_real = next(real_set_iter)
        
        # measure data loading time
        data_time.update(time.time() - end)
        
        # train discriminator first
        batch_size_synthetic = input_tensor_rgb.size(0)
        batch_size_real = input_tensor_rgb_real.size(0)
        
        combined_input_tensors = torch.cat((input_tensor_rgb, input_tensor_rgb_real),dim=0)
        domain_target = torch.cat((torch.zeros(batch_size_synthetic),torch.ones(batch_size_real)),dim=0).to(torch.long)
        with torch.no_grad():
            features_rgb_discriminator = model_hrnet_encoder_rgb(combined_input_tensors)
            outputs_rgb_discriminator = model_rgb(features_rgb_discriminator)
        device = outputs_rgb_discriminator.device
        domain_predicted = model_discriminator(outputs_rgb_discriminator.detach())
        loss_discriminator = discriminator_loss(domain_predicted, domain_target.to(device))
        optimizer_discriminator.zero_grad()
        loss_discriminator.backward()
        optimizer_discriminator.step()
        
        # train generator
        
        # compute output
        features_rgb = model_hrnet_encoder_rgb(combined_input_tensors)
        
        outputs_rgb = model_rgb(features_rgb)

        target_rgb = target_rgb.cuda(non_blocking=True)
        target_weight_rgb = target_weight_rgb.cuda(non_blocking=True)

        output_rgb = outputs_rgb[:batch_size_synthetic]
        output_rgb_real = outputs_rgb[batch_size_synthetic:]
        
        loss_rgb = heatmap_loss_rgb(output_rgb, target_rgb, target_weight_rgb)
        
        domain_predicted = model_discriminator(outputs_rgb)
        
        loss_discriminator = 1-discriminator_loss(domain_predicted, domain_target.to(device))
        
        total_loss = 0.0002*loss_discriminator + loss_rgb
        
        # compute gradient and do update step
        optimizer_rgb.zero_grad()
        
        loss_rgb.backward()
        
        optimizer_rgb.step()

        # measure accuracy and record loss
        _, avg_acc, cnt, pred_real = accuracy(output_rgb_real.detach().cpu().numpy(),
                                              target_rgb_real.detach().cpu().numpy())

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
                      epoch, i, len(train_loader_rgb), batch_time=batch_time,
                      speed=input_tensor_rgb.size(0)/batch_time.val,
                      data_time=data_time, loss_rgb=losses_rgb, acc=acc_rgb)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss_rgb', losses_rgb.val, global_steps)
            writer.add_scalar('train_acc_rgb', acc_rgb.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}_{}'.format(os.path.join(output_dir, 'train'), epoch, i)
            try:
                save_debug_images_uda(
                    config, 
                    input_tensor_rgb, input_tensor_rgb_real,
                    meta, meta_real,
                    pred_rgb*(config.MODEL.IMAGE_SIZE[0]/float(config.MODEL.HEATMAP_SIZE[0])), pred_real*(config.MODEL.IMAGE_SIZE[0]/float(config.MODEL.HEATMAP_SIZE[0])), prefix)
            except:
                print(f'pred_rgb.shape=')
                print(f'pred_real.shape=')
                

def validate_posehrnetfusion_uda(
    config, 
    val_loader, val_dataset,
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

            if i % config.PRINT_FREQ == 0:
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
                save_debug_images_uda_test(
                    config, 
                    input_tensor_rgb, meta, 
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

