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
from utils.vis import save_debug_images_sensor_fusion_eventonly, save_debug_images, save_debug_images_sensor_fusion, save_debug_images_sensor_fusion_overlaymethod
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




def train_posehrnetfusion_nofusion_eventonly(config, 
                        train_loader_event_event,
                        model_event, model_hrnet_encoder_event,
                        heatmap_loss_event,optimizer_event,
                        epoch, output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_event = AverageMeter()
    acc_event = AverageMeter()

    print("TRAINING NO FUSION EVENT ONLY")
    # switch to train mode
    model_hrnet_encoder_event.train()
    model_event.train()
    
    end = time.time()
    for i, (input_tensor_event, target_event, target_weight_event, trans_hm_to_img_event, trans_crop_to_img_event, meta) in enumerate(train_loader_event_event):
        # measure data loading time
        data_time.update(time.time() - end)
        
        features_event = model_hrnet_encoder_event(input_tensor_event)
        outputs_event = model_event(features_event)

        target_event = target_event.cuda(non_blocking=True)
        target_weight_event = target_weight_event.cuda(non_blocking=True)

        output_event = outputs_event

        loss_event = heatmap_loss_event(output_event, target_event, target_weight_event)
        
        # compute gradient and do update step
        
        optimizer_event.zero_grad()
        loss_event.backward()
        optimizer_event.step()

        # measure accuracy and record loss
        losses_event.update(loss_event.item(), input_tensor_event.size(0))

        _, avg_acc, cnt, pred_event = accuracy(output_event.detach().cpu().numpy(),
                                             target_event.detach().cpu().numpy())
        acc_event.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss_event {loss_event.val:.7f} ({loss_event.avg:.7f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader_event_event), batch_time=batch_time,
                      speed=input_tensor_event.size(0)/batch_time.val,
                      data_time=data_time, loss_event=losses_event, acc=acc_event)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss_event', losses_event.val, global_steps)
            writer.add_scalar('train_acc_event', acc_event.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}_{}'.format(os.path.join(output_dir, 'train'), epoch, i)
            save_debug_images_sensor_fusion_eventonly(
                config, 
                input_tensor_event, 
                meta,
                target_event, output_event,
                pred_event*(config.MODEL.IMAGE_SIZE[0]/float(config.MODEL.HEATMAP_SIZE[0])), prefix)

def validate_posehrnetfusion_nofusion_eventonly(
    config, 
    val_loader, 
    val_dataset, 
    model_event, model_hrnet_encoder_event, 
    heatmap_loss_event, 
    output_dir, tb_log_dir, writer_dict=None, epoch=0):
    
    batch_time = AverageMeter()
    losses_event = AverageMeter()
    acc_event = AverageMeter()

    # switch to evaluate mode
    model_event.eval()
    model_hrnet_encoder_event.eval()

    num_samples = len(val_dataset)
    
    keypoint_predictions_event = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    # 4x4 poses
    pose_predictions_event = np.tile(np.eye(4, dtype=np.float32), (num_samples, 1, 1))
    pose_gt_event = np.tile(np.eye(4, dtype=np.float32), (num_samples, 1, 1))
    filenames_event = []

    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input_tensor_event, target_event, target_weight_event, trans_hm_to_img_event, trans_crop_to_img_event, meta) in enumerate(val_loader):
            # compute output
            input_tensor_event = input_tensor_event.cuda()
            features_event = model_hrnet_encoder_event(input_tensor_event)
            # evaluate event outputs
            
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
            
            idx += num_images
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 6 == 0:
                msg = 'Testevent: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses_event, acc=acc_event)
                logger.info(msg)

                prefix = '{}_{}_{}'.format(
                    os.path.join(output_dir, 'validation'), epoch, i
                )
                save_debug_images_sensor_fusion_eventonly(
                    config, 
                    input_tensor_event, meta,
                    target_event, output_event,
                    pred_event*(config.MODEL.IMAGE_SIZE[0]/float(config.MODEL.HEATMAP_SIZE[0])), 
                    prefix)
                

        perf_indicator_event = val_dataset.evaluate(
            config, 
            output_dir,
            pose_gt_event,
            keypoint_predictions_event, 
            pose_predictions_event,
            filenames_event
        )

        # model_name = config.MODEL.NAME
        # if isinstance(name_values, list):
        #     for name_value in name_values:
        #         _print_name_value(name_value, model_name)
        # else:
        #     _print_name_value(name_values, model_name)

    return perf_indicator_event