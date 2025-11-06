import torch
import cv2

from utils.BPnP import BPnP
from kornia.geometry.subpix.spatial_soft_argmax import SpatialSoftArgmax2d

from core.inference import get_final_preds, get_max_preds
from utils.transforms import get_affine_transform


class PoseEstimator:
    def __init__(self, cfg, model, K, landmarks):
        '''
        K: Tensor containing camera intrinsics of shape (3, 3)
        landmarks: Tensor of shape (N, 3)
        '''

        self.model = model
        self.cfg = cfg
        self.K = K
        self.landmarks = landmarks
        self.bpnp = BPnP.apply

    def predict(self, batch, trans_hm_to_img):
        # Predict heatmaps from batch
        # Go from heatmaps to keypoints using DSNT
        # keypoints -> pose using BPnP
        # return: predictions (in image space) as a tuple of (keypoints, poses)

        spatial_softmax = SpatialSoftArgmax2d(normalized_coordinates=False)

        with torch.no_grad():
            outputs = self.model(batch)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            output_shape = output.shape


            # coords_numpy, maxvals = get_final_preds(
            #     self.cfg, output.clone().cpu().numpy(), center.numpy(), scale.numpy())

            # print("numpy:")
            # print(coords_numpy)

            batch_size = output.shape[0]
            num_joints = output.shape[1]
            width = output.shape[3]
            heatmaps_reshaped = output.clone().reshape((batch_size, num_joints, -1))
            maxvals = torch.amax(heatmaps_reshaped, 2)
            maxvals = maxvals.reshape((batch_size, num_joints, 1))

            output = output.mul(255).clamp(0, 255)
            coords = spatial_softmax(output)

            # print("dsnt")
            # print(coords)

            #print("torch")

            # batch_size = output.shape[0]
            # num_joints = output.shape[1]
            # heatmap_width = output.shape[3]
            # heatmaps_reshaped = output.reshape((batch_size, num_joints, -1))
            # idx = torch.argmax(heatmaps_reshaped, dim=2)

            # idx = idx.reshape((batch_size, num_joints, 1))

            # coords_torch = torch.tile(idx, (1, 1, 2)).float()

            # coords_torch[:, :, 0] = (coords_torch[:, :, 0]) % heatmap_width
            # coords_torch[:, :, 1] = torch.floor((coords_torch[:, :, 1]) / heatmap_width)

            # convert the coordinates back to original image space
            heatmap_height = output.shape[2]
            heatmap_width = output.shape[3]

            preds = coords.clone()

            for i in range(coords.shape[0]):
                new_pred = torch.matmul(preds[i], torch.tensor([[1.,0.,0.],[0.,1.,0.]], device='cuda', dtype=torch.float32))
                new_pred += torch.tensor([0.,0.,1.], device='cuda', dtype=torch.float32)
                #trans = get_affine_transform(center[i].cpu().numpy(), scale[i].cpu().numpy(), 0, [heatmap_width, heatmap_height], inv=1)
                trans_torch = trans_hm_to_img[i]
                new_pred = torch.matmul(new_pred, trans_torch.T)
                preds[i] = new_pred

            poses_predicted = self.bpnp(preds, self.landmarks, self.K, maxvals)

            return preds, poses_predicted, maxvals
        
class PoseEstimatorNoModel:
    def __init__(self, cfg, K, landmarks):
        '''
        K: Tensor containing camera intrinsics of shape (3, 3)
        landmarks: Tensor of shape (N, 3)
        camera intrinsics: Tensor of shape (3, 3)
        '''
        self.cfg = cfg
        self.K = K
        self.landmarks = landmarks
        self.bpnp = BPnP.apply

    def predict(self, output, trans_hm_to_img):
        # Predict heatmaps from batch
        # Go from heatmaps to keypoints using DSNT
        # keypoints -> pose using BPnP
        # return: predictions (in image space) as a tuple of (keypoints, poses)

        spatial_softmax = SpatialSoftArgmax2d(normalized_coordinates=False)

        with torch.no_grad():
            batch_size = output.shape[0]
            num_joints = output.shape[1]
            heatmaps_reshaped = output.clone().reshape((batch_size, num_joints, -1))
            maxvals = torch.amax(heatmaps_reshaped, 2)
            maxvals = maxvals.reshape((batch_size, num_joints, 1))

            output = output.mul(255).clamp(0, 255)
            coords = spatial_softmax(output)

            # convert the coordinates back to original image space

            preds = coords.clone()

            for i in range(coords.shape[0]):
                new_pred = torch.matmul(preds[i], torch.tensor([[1.,0.,0.],[0.,1.,0.]], device='cuda', dtype=torch.float32))
                new_pred += torch.tensor([0.,0.,1.], device='cuda', dtype=torch.float32)
                trans_torch = trans_hm_to_img[i]
                new_pred = torch.matmul(new_pred, trans_torch.T)
                preds[i] = new_pred

            poses_predicted = self.bpnp(preds, self.landmarks, self.K, maxvals)

            return preds, poses_predicted, maxvals