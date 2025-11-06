# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torchvision
import cv2

from core.inference import get_max_preds


def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis,
                                 file_name, nrow=8, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]

            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, [255, 0, 0], 2)
            k = k + 1
    cv2.imwrite(file_name, ndarr)

def save_batch_heatmaps_gaussian(batch_image, batch_means_xy, batch_covs_xy, file_name,
                        normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_means_xy: [batch_size, num_joints, 2]
    batch_covs_xy: [batch_size, num_joints, 4, 4]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_means_xy.size(0)
    num_joints = batch_means_xy.size(1)
    heatmap_height = batch_image.size(2)
    heatmap_width = batch_image.size(3)

    extra_cols = 0

    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+extra_cols)*heatmap_width,
                           3),
                          dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            # cv2.circle(masked_image,
            #            (int(preds[i][j][0]), int(preds[i][j][1])),
            #            1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j+extra_cols)
            width_end = heatmap_width * (j+extra_cols+1)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name,
                        normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    extra_cols = 1

    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+extra_cols)*heatmap_width,
                           3),
                          dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            # cv2.circle(masked_image,
            #            (int(preds[i][j][0]), int(preds[i][j][1])),
            #            1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j+extra_cols)
            width_end = heatmap_width * (j+extra_cols+1)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


def save_debug_images(config, input, meta, target, joints_pred, output,
                      prefix):
    if not config.DEBUG.DEBUG:
        return

    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        save_batch_image_with_joints(
            input, meta['joints'], meta['joints_vis'],
            '{}_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_batch_image_with_joints(
            input, joints_pred, meta['joints_vis'],
            '{}_pred.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_HEATMAPS_GT:
        save_batch_heatmaps(
            input, target, '{}_hm_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_HEATMAPS_PRED:
        save_batch_heatmaps(
            input, output, '{}_hm_pred.jpg'.format(prefix)
        )
        
def save_debug_images_sensor_fusion(config, input_event, input_rgb, meta, joints_pred_event, joints_pred_rgb, prefix):
    if not config.DEBUG.DEBUG:
        return

    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        joints_shape = meta['joints_event'].shape
        save_batch_image_with_joints(
            input_event, meta['joints_event'], np.ones((joints_shape[0], joints_shape[1], 1), dtype=np.float32),
            '{}_event_gt.jpg'.format(prefix)
        )
        save_batch_image_with_joints(
            input_rgb, meta['joints_rgb'], np.ones((joints_shape[0], joints_shape[1], 1), dtype=np.float32),
            '{}_rgb_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_batch_image_with_joints(
            input_event, joints_pred_event, np.ones((joints_shape[0], joints_shape[1], 1), dtype=np.float32),
            '{}__event_pred.jpg'.format(prefix)
        )
        save_batch_image_with_joints(
            input_rgb, joints_pred_rgb, np.ones((joints_shape[0], joints_shape[1], 1), dtype=np.float32),
            '{}__rgb_pred.jpg'.format(prefix)
        )

def save_debug_images_sensor_fusion_rgbonly(config, input_rgb, meta, target_rgb, output_rgb, joints_pred_rgb, prefix):
    if not config.DEBUG.DEBUG:
        return

    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        joints_shape = meta['joints_rgb'].shape
        save_batch_image_with_joints(
            input_rgb, meta['joints_rgb'], np.ones((joints_shape[0], joints_shape[1], 1), dtype=np.float32),
            '{}_rgb_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_batch_image_with_joints(
            input_rgb, joints_pred_rgb, np.ones((joints_shape[0], joints_shape[1], 1), dtype=np.float32),
            '{}_rgb_pred.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_HEATMAPS_GT:
        save_batch_heatmaps(
            input_rgb, target_rgb, '{}_rgb_hm_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_HEATMAPS_PRED:
        save_batch_heatmaps(
            input_rgb, output_rgb, '{}_rgb_hm_pred.jpg'.format(prefix)
        )
 
def save_debug_images_uda(config, input_rgb, input_rgb_real, meta, meta_real, joints_pred_rgb, joints_pred_rgb_real, prefix):
    if not config.DEBUG.DEBUG:
        return

    joints_shape = meta['joints_rgb'].shape
    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        save_batch_image_with_joints(
            input_rgb, meta['joints_rgb'], np.ones((joints_shape[0], joints_shape[1], 1), dtype=np.float32),
            '{}_rgb_gt.jpg'.format(prefix)
        )
        save_batch_image_with_joints(
            input_rgb_real, meta_real['joints_rgb'], np.ones((joints_shape[0], joints_shape[1], 1), dtype=np.float32),
            '{}_rgb_real_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_batch_image_with_joints(
            input_rgb, joints_pred_rgb, np.ones((joints_shape[0], joints_shape[1], 1), dtype=np.float32),
            '{}__rgb_pred.jpg'.format(prefix)
        )
        save_batch_image_with_joints(
            input_rgb_real, joints_pred_rgb_real, np.ones((joints_shape[0], joints_shape[1], 1), dtype=np.float32),
            '{}__rgb_real_pred.jpg'.format(prefix)
        )
        
def save_debug_images_uda_test(config, input_rgb, meta, joints_pred_rgb, prefix):
    if not config.DEBUG.DEBUG:
        return

    joints_shape = meta['joints_rgb'].shape
    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        save_batch_image_with_joints(
            input_rgb, meta['joints_rgb'], np.ones((joints_shape[0], joints_shape[1], 1), dtype=np.float32),
            '{}_rgb_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_batch_image_with_joints(
            input_rgb, joints_pred_rgb, np.ones((joints_shape[0], joints_shape[1], 1), dtype=np.float32),
            '{}__rgb_pred.jpg'.format(prefix)
        )

def save_debug_images_certifier_test(config, input_event, meta, joints_pred_event, prefix):
    if not config.DEBUG.DEBUG:
        return

    joints_shape = meta['joints_event'].shape
    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        save_batch_image_with_joints(
            input_event, meta['joints_event'], np.ones((joints_shape[0], joints_shape[1], 1), dtype=np.float32),
            '{}_event_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_batch_image_with_joints(
            input_event, joints_pred_event, np.ones((joints_shape[0], joints_shape[1], 1), dtype=np.float32),
            '{}__event_pred.jpg'.format(prefix)
        )
        
# def save_debug_images_certifier(config, input_event, meta, joints_pred_event, prefix):
#     if not config.DEBUG.DEBUG:
#         return

#     joints_shape = meta['joints_event'].shape
#     if config.DEBUG.SAVE_BATCH_IMAGES_GT:
#         save_batch_image_with_joints(
#             input_event, meta['joints_event'], np.ones((joints_shape[0], joints_shape[1], 1), dtype=np.float32),
#             '{}_event_gt.jpg'.format(prefix)
#         )
#     if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
#         save_batch_image_with_joints(
#             input_event, joints_pred_event, np.ones((joints_shape[0], joints_shape[1], 1), dtype=np.float32),
#             '{}__event_pred.jpg'.format(prefix)
#         )

def save_debug_images_sensor_fusion_overlaymethod(config, input_event, input_rgb, combined_tensor, meta, joints_pred_event, joints_pred_rgb, prefix):
    if not config.DEBUG.DEBUG:
        return

    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        joints_shape = meta['joints_event'].shape
        save_batch_image_with_joints(
            input_event, meta['joints_event'], np.ones((joints_shape[0], joints_shape[1], 1), dtype=np.float32),
            '{}_event_gt.jpg'.format(prefix)
        )
        save_batch_image_with_joints(
            input_rgb, meta['joints_rgb'], np.ones((joints_shape[0], joints_shape[1], 1), dtype=np.float32),
            '{}_rgb_gt.jpg'.format(prefix)
        )
        save_batch_image_with_joints(
            combined_tensor, meta['joints_event'], np.ones((joints_shape[0], joints_shape[1], 1), dtype=np.float32),
            '{}_combined_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_batch_image_with_joints(
            input_event, joints_pred_event, np.ones((joints_shape[0], joints_shape[1], 1), dtype=np.float32),
            '{}__event_pred.jpg'.format(prefix)
        )
        save_batch_image_with_joints(
            input_rgb, joints_pred_rgb, np.ones((joints_shape[0], joints_shape[1], 1), dtype=np.float32),
            '{}__rgb_pred.jpg'.format(prefix)
        )
        save_batch_image_with_joints(
            combined_tensor, joints_pred_event, np.ones((joints_shape[0], joints_shape[1], 1), dtype=np.float32),
            '{}__combined_pred.jpg'.format(prefix)
        )

# def save_debug_images_certifier(config, input, meta, target, joints_pred, joints_pred_corrected, output, prefix):
#     if not config.DEBUG.DEBUG:
#         return

#     if config.DEBUG.SAVE_BATCH_IMAGES_GT:
#         save_batch_image_with_joints(
#             input, meta['joints'], meta['joints_vis'],
#             '{}_gt.jpg'.format(prefix)
#         )
#     if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
#         save_batch_image_with_joints(
#             input, joints_pred, meta['joints_vis'],
#             '{}_pred.jpg'.format(prefix)
#         )
#         save_batch_image_with_joints(
#             input, joints_pred_corrected, meta['joints_vis'],
#             '{}_pred_corrected.jpg'.format(prefix)
#         )
        
def save_debug_images_certifier(config, input, meta, target, joints_pred, joints_pred_corrected, output, prefix):
    if not config.DEBUG.DEBUG:
        return

    joints_shape = meta['joints_event'].shape
    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        save_batch_image_with_joints(
            input, meta['joints_event'], np.ones((joints_shape[0], joints_shape[1], 1)),
            '{}_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_batch_image_with_joints(
            input, joints_pred, np.ones((joints_shape[0], joints_shape[1], 1)),
            '{}_pred.jpg'.format(prefix)
        )
        save_batch_image_with_joints(
            input, joints_pred_corrected, np.ones((joints_shape[0], joints_shape[1], 1)),
            '{}_pred_corrected.jpg'.format(prefix)
        )

