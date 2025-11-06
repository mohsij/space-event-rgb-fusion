import json
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import torch
import math
from tqdm import tqdm
import time

import argparse

from kornia.geometry.conversions import rotation_matrix_to_quaternion

def parse_args():
    parser = argparse.ArgumentParser(description='generate plots')
    # general
    parser.add_argument('--scene',
                        help='scene name',
                        required=True,
                        type=str)

    parser.add_argument('--data_dir',
                        help='base data dir',
                        required=True,
                        type=str)

    args = parser.parse_args()

    return args

def plot_wireframe(img, wireframe_faces, projected_wireframe_points, c=(0,255,0)):
    isClosed = True
    color = c
    thickness = 1

    for face in wireframe_faces:
        pts = np.array([projected_wireframe_points[face_point] for face_point in face], dtype=np.int32)
        img = cv2.polylines(img, [pts], isClosed, color, thickness)
    return img

def main():
    args = parse_args()
    scene = args.scene
    data_dir = args.data_dir
    print(scene)
    satellite = scene.split('-')[0]

    plot_uncertainty = False

    uncertainty_scale = [0,200]
    ransaciters = 10000

    reprojection_error_thresh = 20

    predictions_file = 'new-results/{}/EventRGBAlignedDataset/pose_hrnet_dropout/{}/predictions_fusion.json'.format(satellite, scene)

    predictions_file_nodropout = 'new-results/{}/EventRGBAlignedDataset/pose_hrnet_nodropout/{}/predictions_nofusion.json'.format(satellite, scene)

    with open(predictions_file, 'r') as jsonfile:
        predictions = json.load(jsonfile)
        
    with open(predictions_file_nodropout, 'r') as jsonfile:
        predictions_nodropout = json.load(jsonfile)

    with open(predictions_file_nodropout, 'r') as jsonfile:
        predictions_fusion = json.load(jsonfile)

    # gt file
    gt_dir = data_dir
    with open(os.path.join('../../data/new-data/', gt_dir, scene, 'test_event_detectron_estimate.json'), 'r') as jsonfile:
        gt_labels_all = json.load(jsonfile)

    data_path = os.path.join('../../data/new-data/', gt_dir, scene, 'frames')    

    uncertainty_computation_time = time.time()

    keypoints_rgb = np.array([predictions['annotations'][image_index]['keypoints_{}'.format('rgb')] for image_index in range(len(predictions['annotations']))])
    keypoints_event = np.array([predictions['annotations'][image_index]['keypoints_{}'.format('event')] for image_index in range(len(predictions['annotations']))])
    keypoints_nodropout_rgb = np.array([predictions_nodropout['annotations'][image_index]['keypoints_{}'.format('rgb')] for image_index in range(len(predictions['annotations']))])
    keypoints_nodropout_event = np.array([predictions_nodropout['annotations'][image_index]['keypoints_{}'.format('event')] for image_index in range(len(predictions['annotations']))])

    # print(keypoints_rgb.shape)
    # print(keypoints_nodropout_rgb.shape)

    gt_keypoints = np.array([gt_labels_all['annotations'][image_index]['keypoints'] for image_index in range(len(predictions['annotations']))])

    # for each keypoint calculate uncertainty (i.e 2x standard deviation)
    # to get (images, 18, 1) array of uncertainty values for event and rgb separately
    uncertainties_rgb = np.mean(np.std(keypoints_rgb.transpose(0,1,3,2), axis=-1) * 2, axis=-1)
    median_uncertainties_rgb = np.median(uncertainties_rgb, axis=-1)
    # uncertainties_rgb = np.clip(uncertainties_rgb, uncertainty_scale[0], uncertainty_scale[1])

    uncertainties_event = np.mean(np.std(keypoints_event.transpose(0,1,3,2), axis=-1) * 2, axis=-1)
    median_uncertainties_event = np.median(uncertainties_event, axis=-1)
    # uncertainties_event = np.clip(uncertainties_event, uncertainty_scale[0], uncertainty_scale[1])

    instance_count = len(gt_labels_all['annotations'])

    uncertainty_computation_time = time.time() - uncertainty_computation_time
    uncertainty_computation_time /= instance_count

    print('total instance count: {}'.format(instance_count))
    print(f'{uncertainty_computation_time=}')

    print(f'{uncertainties_rgb.shape=}')
    print(np.median(uncertainties_event))
    print(np.min(uncertainties_event))
    print(np.median(uncertainties_rgb))
    print(np.min(uncertainties_rgb))

    avg_time_ransac_only = 0.0
    avg_time_full_method = 0.0

    count_ransac_only = 0
    count_full_method = 0

    landmarks_3d = np.array(gt_labels_all['landmarks_3d'])

    landmark_count = landmarks_3d.shape[0]

    distortion_coefficients = np.array([0.,0.,0.,0.,0.])

    wireframe_points = np.array(gt_labels_all['wireframe_points'])
    wireframe_faces = gt_labels_all['wireframe_faces']

    # then run pnp to get poses in the rgb coordinate frame
    # first with just regular pnp and ransac separately for event and rgb points
    # then with combined pnp where you drop points first based on uncertainty > 10px
    K_rgb = np.array(gt_labels_all['intrinsics'])
    K_4x4_rgb = np.eye(4)
    K_4x4_rgb[0:3,0:3] = K_rgb

    K_event = np.array(gt_labels_all['intrinsics'])
    K_4x4_event = np.eye(4)
    K_4x4_event[0:3,0:3] = K_event

    for image_index_single in tqdm(range(len(gt_labels_all['annotations']))):
        start_time = time.time()
        is_full_method = False

        keypoints_single_rgb = keypoints_nodropout_rgb[image_index_single,:,:]
        keypoints_single_event = keypoints_nodropout_event[image_index_single,:,:]

        bbox = gt_labels_all['annotations'][image_index_single]['bbox'][:4]
        x1, y1, x2, y2 = bbox[:4]
        w = x2 - x1
        h = y2 - y1

        bbox_diagonal = np.sqrt(float(w*w) + float(h*h))

        keypoint_distances = []
        for j in range(landmark_count):
            a = keypoints_single_rgb[j]
            b = keypoints_single_event[j]
            keypoint_distances.append(np.linalg.norm(a-b))

        # 4th method uncertainty based decision
        keypoints_filtered = np.vstack((keypoints_nodropout_rgb[image_index_single,:,:],
                                       keypoints_nodropout_event[image_index_single,:,:]))
        landmarks_filtered = np.vstack((landmarks_3d, landmarks_3d))

        # the threshold should also be according to bbox
        keypoint_distance_thresh = (bbox_diagonal * 0.2)
        # print(bbox_diagonal)
        if np.median(keypoint_distances) > keypoint_distance_thresh:
            # event channel is better according to uncertainty
            if median_uncertainties_event[image_index_single] < median_uncertainties_rgb[image_index_single]:
                keypoints_filtered = keypoints_nodropout_event[image_index_single,:,:]
                landmarks_filtered = landmarks_3d
            # rgb channel is better
            else:
                keypoints_filtered = keypoints_nodropout_rgb[image_index_single,:,:]
                landmarks_filtered = landmarks_3d
            is_full_method = True


        # uncertainties_switch = np.argmin(uncertainties_combined, axis=-1)
        # min_uncertainties = np.min(uncertainties_combined, axis=-1)
        # keypoints_filtered = keypoints_filtered[range(uncertainties_switch.shape[0]),uncertainties_switch,:]

        ret, pred_rotation_vector, pred_translation, inliers_filtered = cv2.solvePnPRansac(
            landmarks_filtered, keypoints_filtered, K_rgb,
            flags=cv2.SOLVEPNP_EPNP, iterationsCount=ransaciters, reprojectionError=reprojection_error_thresh, distCoeffs=distortion_coefficients)

        time_taken = time.time() - start_time
        if is_full_method:
            avg_time_full_method += time_taken
            avg_time_full_method += uncertainty_computation_time
            count_full_method += 1
        else:
            avg_time_ransac_only += time_taken
            count_ransac_only += 1


        pred_rotation_matrix, _ = cv2.Rodrigues(pred_rotation_vector)
        rt = np.column_stack((pred_rotation_matrix, pred_translation))
        full_pose_filtered = np.eye(4)
        full_pose_filtered[0:3,:] = rt

        predictions_fusion['annotations'][image_index_single]['pose_rgb'] = full_pose_filtered.tolist()
        predictions_fusion['annotations'][image_index_single]['pose_event'] = full_pose_filtered.tolist()
    
    avg_time_full_method /= count_full_method
    avg_time_ransac_only /= count_ransac_only

    print(f'{avg_time_full_method=}')
    print(f'{avg_time_ransac_only=}')
    print(f'{count_full_method=}')
    print(f'{count_ransac_only=}')
    # with open(predictions_file_nodropout.replace('predictions_nofusion', 'predictions_ourmethod_halfbbox_cmkd'), 'w') as jsonfile:
    #     jsonfile.write(json.dumps(predictions_fusion, indent=2))



if __name__ == '__main__':
    main()