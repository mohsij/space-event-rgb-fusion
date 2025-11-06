# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import torch

import torchvision.transforms.functional as F
from torchvision import transforms

EVENT_GRAY_TENSOR_FILL = 0.4980

def flip_back(output_flipped, matched_parts):
    '''
    ouput_flipped: numpy.ndarray(batch_size, num_joints, height, width)
    '''
    assert output_flipped.ndim == 4,\
        'output_flipped should be [batch_size, num_joints, height, width]'

    output_flipped = output_flipped[:, :, :, ::-1]

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0], :, :].copy()
        output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
        output_flipped[:, pair[1], :, :] = tmp

    return output_flipped


def fliplr_joints(joints, joints_vis, width, matched_parts):
    """
    flip coords
    """
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = \
            joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return joints*joints_vis, joints_vis


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_affine_transform(
        center, scale, rot, output_size,
        shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print("converting this scale: ", scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(
        img, trans, (int(output_size[0]), int(output_size[1])),
        flags=cv2.INTER_LINEAR
    )

    return dst_img

def BlendTransform(img, src_image, src_weight, dst_weight):
    if img.dtype == np.uint8:
        img = img.astype(np.float32)
        img = src_weight * src_image + dst_weight * img
        return np.clip(img, 0, 255).astype(np.uint8)
    else:
        return src_weight * src_image + dst_weight * img

class ToNumpy(object):
    def __init__(self):
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """

    def __call__(self, image):
        img = np.array(image)
        return img


### Augmentations created for speedplus ###
class RandomHaze(object):
    """
    Add random gaussian noise to an image.
    """
    def __init__(self, mean_min=0.05, mean_max=0.15, std_min=0.03, std_max=0.05):
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """

        self.std_min = std_min
        self.std_range = std_max - std_min
        self.mean_min = mean_min
        self.mean_range = mean_max - mean_min

    def _translate_image(self, image, W=1920, H=1200, t_x=0, t_y=0):
        M = np.float32([[1, 0, t_x], 
                        [0, 1, t_y],
                        [0, 0, 1]])
        return cv2.warpPerspective(image, M, (W, H))
        
    def _scale_image(self, image, W=1920, H=1200, s_x=0, s_y=0):
        M = np.float32([[1.5, 0, 0], 
                        [0, 1.8, 0],
                        [0, 0, 1]])
        return cv2.warpPerspective(image, M, (W, H))

    def __call__(self, image):
        H, W, C = image.shape
        assert image.min()>=0

        noise = np.random.randn(H, W, 1).repeat(3, -1)

        white_pixel = 255
        std = np.random.rand(1)*self.std_range + self.std_min
        mean = np.random.rand(1)*self.mean_range + self.mean_min
        noise = (white_pixel*std)*(white_pixel*noise)+(white_pixel*mean)
        noise = noise.clip(min=0., max=255.)
        noise = np.array(Image.fromarray(np.uint8(noise)).filter(ImageFilter.GaussianBlur(radius=5)))

        noise[noise < np.random.randint(125,140)] = 0
        noise = self._scale_image(noise, s_x=np.random.randint(0.75, 1.25), s_y=np.random.randint(0.75, 1.25), W=W, H=H)
        noise = np.array(Image.fromarray(np.uint8(noise)).filter(ImageFilter.GaussianBlur(radius=np.random.uniform(25,40))))

        # either overlay or don't
        w = np.random.uniform(0.1, 0.8)
        return BlendTransform(image, src_image=noise, src_weight=w, dst_weight=1)

class RandomFlares(object):
    """
    Add random gaussian noise to an image.
    """
    def __init__(self):
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """


    def _rotate_image(self, image, angle, W=1920, H=1200):
        image_center = (W/2, H/2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def _polygon_vertices(self, x, y, r, sides=6):
        vertices = [[x, y + r]]
        for angle in np.linspace(0, 2 * np.pi, sides):
            vertices.append([x + r * np.sin(angle), y + r * np.cos(angle)])
        vertices = np.array(vertices, dtype=np.int32)
        return vertices

    def _shear_image(self, image, W=1920, H=1200, scale=0):
        M = np.float32([[1, scale, 0], 
                        [0, 1, 0],
                        [0, 0, 1]])
        return cv2.warpPerspective(image, M, (W, H))
        
    def __call__(self, image):
        H, W, C = image.shape
        assert image.min()>=0.
        #assert image.max()<=1.

        blank = np.full((H, W, 1), 0, dtype=np.uint8).repeat(3, -1)

        for i in range(np.random.randint(1, 10)):
            centre_x = 1920/2
            centre_y = 1200/2
            x_offset = np.random.randint(centre_x - 500, centre_x + 500)
            y_offset = np.random.randint(centre_y - 400, centre_y + 400)
            # make a pentagon
            pts = self._polygon_vertices(x_offset, y_offset, np.random.randint(5,100))
     
            color = (255, 255, 255)
            cv2.fillPoly(blank, [pts], color)
            blank  = self._rotate_image(blank, np.random.randint(0, 180), W=W, H=H)
            blank = self._shear_image(blank, scale=np.random.uniform(0, 0.75), W=W, H=H)
            blank = np.array(Image.fromarray(np.uint8(blank)).filter(ImageFilter.GaussianBlur(radius=np.random.uniform(1,5))))
            blank = blank * np.random.uniform(0.4, 1.2)

        # either overlay or don't
        w = np.random.uniform(0, 1)
        return BlendTransform(image, src_image=blank, src_weight=w, dst_weight=1)

class RandomStreaks(object):
    """
    Add random gaussian noise to an image.
    """
    def __init__(self, mean_min=0.05, mean_max=0.15, std_min=0.03, std_max=0.05):
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """

        self.std_min = std_min
        self.std_range = std_max - std_min
        self.mean_min = mean_min
        self.mean_range = mean_max - mean_min

    def _rotate_image(self, image, angle, W=1920, H=1200):
        rot_mat = cv2.getRotationMatrix2D((W/2, H/2), angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def _translate_image(self, image, t_x=0, t_y=0, W=1920, H=1200):
        M = np.float32([[1, 0, t_x], 
                        [0, 1, t_y],
                        [0, 0, 1]])
        return cv2.warpPerspective(image, M, (W, H))
        
    def _scale_image(self, image, s_x=1, s_y=1, W=1920, H=1200):
        M = np.float32([[s_x, 0, 0], 
                        [0, s_y, 0],
                        [0, 0, 1]])
        return cv2.warpPerspective(image, M, (W, H))

    def _shear_image(self, image, scale=0, W=1920, H=1200):
        M = np.float32([[1, scale, 0], 
                        [0, 1, 0],
                        [0, 0, 1]])
        return cv2.warpPerspective(image, M, (W, H))

    def _motion_blur(self, image, kernel_size=15):
        kernel_motion_blur = np.zeros((size, size))
        kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size

        # applying the kernel to the input image
        return cv2.filter2D(image, -1, kernel_motion_blur)

    def _radial_blur(self, image, blur_amount=0.01, iterations=5, center_x=0, center_y=0):
        # From : https://stackoverflow.com/questions/7607464/implement-radial-blur-with-opencv
        blur = blur_amount

        w, h = image.shape[:2]

        growMapx = np.tile(np.arange(h) + ((np.arange(h) - center_x)*blur), (w, 1)).astype(np.float32)
        shrinkMapx = np.tile(np.arange(h) - ((np.arange(h) - center_x)*blur), (w, 1)).astype(np.float32)
        growMapy = np.tile(np.arange(w) + ((np.arange(w) - center_y)*blur), (h, 1)).transpose().astype(np.float32)
        shrinkMapy = np.tile(np.arange(w) - ((np.arange(w) - center_y)*blur), (h, 1)).transpose().astype(np.float32)
        growMapx, growMapy = np.abs(growMapx), np.abs(growMapy)
        for i in range(iterations):
            tmp1 = cv2.remap(image, growMapx, growMapy, cv2.INTER_LINEAR)
            tmp2 = cv2.remap(image, shrinkMapx, shrinkMapy, cv2.INTER_LINEAR)
            image = cv2.addWeighted(tmp1, 0.5, tmp2, 0.5, 0)
        return image

    def _radial_fade(self, image, W=1920, H=1200):
        # From: https://stackoverflow.com/questions/62045155/how-to-create-a-transparent-radial-gradient-with-python

        # Create radial alpha/transparency layer. 255 in centre, 0 at edge
        X = np.linspace(-1, 1, H)[:, None]*255
        Y = np.linspace(-1, 1, W)[None, :]*255
        alpha = np.sqrt(X**2 + Y**2)
        alpha = 255 - np.clip(0,255,alpha)
        alpha = np.expand_dims(alpha, -1)
        alpha = alpha.repeat(3, -1)
        # Push that radial gradient transparency onto red image and save
        #return Image.fromarray(image.astype(np.uint8)).putalpha(Image.fromarray(alpha.astype(np.uint8)))
        return image * (alpha / 255)

    def __call__(self, image):
        H, W, C = image.shape
        assert image.min()>=0.
        #assert image.max()<=1.

        noise = np.random.randn(H, W, 1).repeat(3, -1)

        white_pixel = 255
        std = np.random.rand(1)*self.std_range + self.std_min
        mean = np.random.rand(1)*self.mean_range + self.mean_min
        noise = (white_pixel*std)*(white_pixel*noise)+(white_pixel*mean)
        noise = noise.clip(min=0., max=255.)
        noise = np.array(Image.fromarray(np.uint8(noise)).filter(ImageFilter.GaussianBlur(radius=1)))
        noise[noise < np.random.randint(150,200)] = 0
        noise = self._radial_blur(noise, np.random.uniform(0.01, 0.04), 5, np.random.randint(0,1920), np.random.randint(0,1200))
        noise = self._radial_fade(noise, W, H)
        if np.random.randint(2) == 1:
            noise = self._scale_image(noise, s_x=np.random.uniform(2, 4), W=W, H=H)
        else:
            noise = self._scale_image(noise, s_y=np.random.uniform(2, 4), W=W, H=H)

        noise = self._rotate_image(noise, np.random.uniform(0, np.pi), W=W, H=H)
        # either overlay or don't
        w = np.random.uniform(0, 1)
        return BlendTransform(image, src_image=noise, src_weight=w, dst_weight=1)

class RandomBloom(object):
    """
    Add random gaussian noise to an image.
    """
    def __init__(self):
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """

    def __call__(self, image):
        H, W, C = image.shape
        
        offset = np.random.randint(10, 100)

        if not offset % 2 == 0:
            offset += 1

        bloom_image = np.copy(image)
        bloom_image = cv2.GaussianBlur(bloom_image, ksize=(9 + offset, 9 + offset), sigmaX=10, sigmaY=10)
        bloom_image = cv2.blur(bloom_image, ksize=(5 + offset, 5 + offset))

        offset = np.random.randint(0, 200)

        w = 1
        return BlendTransform(image, src_image=bloom_image, src_weight=w, dst_weight=1)

class RandomBlur(object):
    """
    Add random gaussian noise to an image.
    """
    def __init__(self):
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """

    def __call__(self, image):
        blur_kernel = [np.random.randint(1,10), np.random.randint(1,10)]
        if blur_kernel[0] % 2 == 0:
            blur_kernel[0] += 1
        if blur_kernel[1] % 2 == 0:
            blur_kernel[1] += 1
        return transforms.GaussianBlur(kernel_size=blur_kernel, sigma=(0.1,5.))(image)

    
class RandomUniformNoise(torch.nn.Module):
    def __init__(self, mean_min=0.01, mean_max=0.15, std_min=0.01, std_max=0.05):
        super().__init__()
        self.std_min = std_min
        self.std_range = std_max - std_min
        self.mean_min = mean_min
        self.mean_range = mean_max - mean_min
    def forward(self, img):
        C, H, W = img.size()
        img.clamp(min=0., max=1.)

        noise = torch.randn(1,H,W).repeat(3,1,1)
        
        std = torch.rand(1)*self.std_range + self.std_min
        mean = torch.rand(1)*self.mean_range + self.mean_min
        img = std*noise+mean+img
        return img.clamp(min=0., max=1.)
    
class RandomColourNoise(torch.nn.Module):
    def __init__(self, mean_min=0.01, mean_max=0.1, std_min=0.01, std_max=0.05):
        super().__init__()
        self.std_min = std_min
        self.std_range = std_max - std_min
        self.mean_min = mean_min
        self.mean_range = mean_max - mean_min
    def forward(self, img):
        C, H, W = img.size()
        img.clamp(min=0., max=1.)

        noise = torch.randn(C,H,W)
        
        std = torch.rand(1)*self.std_range + self.std_min
        mean = torch.rand(1)*self.mean_range + self.mean_min
        img = std*noise+mean+img
        return img.clamp(min=0., max=1.)

class RandomBars(torch.nn.Module):
    def __init__(self, fill_value=EVENT_GRAY_TENSOR_FILL):
        super().__init__()
        self._fill_value = fill_value
    def forward(self, img):
        C, H, W = img.size()
        
        bar_size = 0.1
        # top
        img[:C,0:np.random.randint(low=0,high=int(H*bar_size)),:] = self._fill_value
        # bot
        img[:C,np.random.randint(low=int((1-bar_size)*H),high=H):H,:] = self._fill_value
        
        # left
        img[:C,:,0:np.random.randint(low=0,high=int(W*bar_size))] = self._fill_value
        # right
        img[:C,:,np.random.randint(low=int((1-bar_size)*W),high=W):W] = self._fill_value

        return img

def interp(values, leftMin, leftMax, rightMin, rightMax):
    leftSpan = float(leftMax) - float(leftMin)
    rightSpan = float(rightMax) - float(rightMin)
    return rightMin+(((values-leftMin)/leftSpan)*rightSpan)

class FillEventBlack(torch.nn.Module):
    """Add random white noise to the event frame
    """

    def __init__(self):
        super().__init__()

    def forward(self, img):
        """
        noise_remove = np.interp(np.clip(np.random.rand(*img.shape), 0, percent_to_change), [0, percent_to_change], [0, 1])
        noise_add = np.interp(np.clip(np.random.rand(*img.shape), 1-percent_to_change, 1), [1-percent_to_change, 1], [1, 2])
        img_aug = np.round(np.clip(img*noise_remove, 127, 255)).astype(np.uint8)
        img_aug = np.round(np.clip(img_aug*noise_add, 127, 255)).astype(np.uint8)
        Args:
            img (PIL Image or Tensor): Image to be noised.

        Returns:
            PIL Image or Tensor: noised image.
        """
        return torch.clamp(img, EVENT_GRAY_TENSOR_FILL, 1) # 127


    def __repr__(self) -> str:
        format_string = self.__class__.__name__
        return format_string
    
class FlipBlackEventsToWhite(torch.nn.Module):
    """Change negative polarity events to positive polarity
    """

    def __init__(self):
        super().__init__()

    def forward(self, img):
        """
        noise_remove = np.interp(np.clip(np.random.rand(*img.shape), 0, percent_to_change), [0, percent_to_change], [0, 1])
        noise_add = np.interp(np.clip(np.random.rand(*img.shape), 1-percent_to_change, 1), [1-percent_to_change, 1], [1, 2])
        img_aug = np.round(np.clip(img*noise_remove, 127, 255)).astype(np.uint8)
        img_aug = np.round(np.clip(img_aug*noise_add, 127, 255)).astype(np.uint8)
        Args:
            img (PIL Image or Tensor): Image to be noised.

        Returns:
            PIL Image or Tensor: noised image.
        """
        img = img - EVENT_GRAY_TENSOR_FILL
        img = torch.absolute(img)
        img = img + EVENT_GRAY_TENSOR_FILL
        return torch.clamp(img, EVENT_GRAY_TENSOR_FILL, 1) # 127


    def __repr__(self) -> str:
        format_string = self.__class__.__name__
        return format_string
    
class EventNormalise(torch.nn.Module):
    """Add random white noise to the event frame
    """

    def __init__(self, normval=0.5):
        super().__init__()
        self.normval = normval

    def forward(self, img):
        """
        noise_remove = np.interp(np.clip(np.random.rand(*img.shape), 0, percent_to_change), [0, percent_to_change], [0, 1])
        noise_add = np.interp(np.clip(np.random.rand(*img.shape), 1-percent_to_change, 1), [1-percent_to_change, 1], [1, 2])
        img_aug = np.round(np.clip(img*noise_remove, 127, 255)).astype(np.uint8)
        img_aug = np.round(np.clip(img_aug*noise_add, 127, 255)).astype(np.uint8)
        Args:
            img (PIL Image or Tensor): Image to be noised.

        Returns:
            PIL Image or Tensor: noised image.
        """
        img[img > self.normval] = 1
        return img


    def __repr__(self) -> str:
        format_string = self.__class__.__name__
        return format_string
    
class MakeEventsColoured(torch.nn.Module):
    """Add random white noise to the event frame
    """

    def __init__(self):
        super().__init__()

    def forward(self, img):
        """
        noise_remove = np.interp(np.clip(np.random.rand(*img.shape), 0, percent_to_change), [0, percent_to_change], [0, 1])
        noise_add = np.interp(np.clip(np.random.rand(*img.shape), 1-percent_to_change, 1), [1-percent_to_change, 1], [1, 2])
        img_aug = np.round(np.clip(img*noise_remove, 127, 255)).astype(np.uint8)
        img_aug = np.round(np.clip(img_aug*noise_add, 127, 255)).astype(np.uint8)
        Args:
            img (PIL Image or Tensor): Image to be noised.

        Returns:
            PIL Image or Tensor: noised image.
        """
        img[0,:,:] = 0
        img[2,:,:] = 0
        return img


    def __repr__(self) -> str:
        format_string = self.__class__.__name__
        return format_string

class RandomEventNoise(torch.nn.Module):
    """Add random white noise to the event frame
    """

    def __init__(self, brighten=True):
        super().__init__()
        self._brighten = brighten

    def forward(self, img):
        """
        noise_remove = np.interp(np.clip(np.random.rand(*img.shape), 0, percent_to_change), [0, percent_to_change], [0, 1])
        noise_add = np.interp(np.clip(np.random.rand(*img.shape), 1-percent_to_change, 1), [1-percent_to_change, 1], [1, 2])
        img_aug = np.round(np.clip(img*noise_remove, 127, 255)).astype(np.uint8)
        img_aug = np.round(np.clip(img_aug*noise_add, 127, 255)).astype(np.uint8)
        Args:
            img (PIL Image or Tensor): Image to be noised.

        Returns:
            PIL Image or Tensor: noised image.
        """
        min_val = img.min()
        max_val = img.max()
        if not isinstance(img, torch.Tensor):
            raise(TypeError('Event noise can only be applied to tensors'))
        if self._brighten:
            percent_to_brighten = float(torch.empty(1).uniform_(float(0.001), float(0.2)).item())
            noise_brighten = interp(torch.clamp(torch.rand(img.shape), 1-percent_to_brighten, 1), 1-percent_to_brighten, 1, 1, 2.5)
            return torch.clamp((img*noise_brighten), EVENT_GRAY_TENSOR_FILL, 1)
        else:
            percent_to_darken = float(torch.empty(1).uniform_(float(0.1), float(0.9)).item())
            noise_darken = interp(torch.clamp(torch.rand(img.shape), 0, percent_to_darken), 0, percent_to_darken, 0, 1)
            return torch.clamp((img*noise_darken), EVENT_GRAY_TENSOR_FILL, 1)


    def __repr__(self) -> str:
        format_string = self.__class__.__name__
        return format_string

class RandomEventPatchNoise(torch.nn.Module):
    """Add a random patch of white noise to the event frame
       Get a white noise tensor then apply RandomPerspective->RandomAffine->
    """

    def __init__(self, brighten=True):
        super().__init__()
        self._brighten = brighten
        mask_fill = 0 if self._brighten else 1
        self._blur_kernel = (3,3)
        self._blur_sigma = (0.2, 0.7)
        self._shapeTransforms = transforms.Compose([
                transforms.RandomAffine(45, translate=(0.1,0.5),scale=(0.1,0.8), shear=90, fill=mask_fill)])

    def forward(self, img):
        """
        noise_add = np.interp(np.clip(np.random.rand(*img.shape), 1-percent_to_change, 1), [1-percent_to_change, 1], [1, 2])
        img_aug = np.round(np.clip(img*noise_add, 127, 255)).astype(np.uint8)
        Args:
            img (PIL Image or Tensor): Image to be noised.

        Returns:
            PIL Image or Tensor: noised image.
        """
        percent_to_brighten = float(torch.empty(1).uniform_(float(0.001), float(0.1)).item())
        percent_to_darken = float(torch.empty(1).uniform_(float(0.1), float(0.9)).item())
        if self._brighten:
            noise = interp(torch.clamp(torch.rand(img.shape), 1-percent_to_brighten, 1), 1-percent_to_brighten, 1, 1, 2)
            mask = torch.ones_like(noise)
            transformed_mask = self._shapeTransforms(mask)
            masked_noise = torch.clamp(transformed_mask*noise, 1, 2)
            masked_noise = transforms.GaussianBlur(kernel_size=self._blur_kernel, sigma=self._blur_sigma)(masked_noise)
            masked_noise = torch.clamp(masked_noise, 1, 2)
            res = torch.clamp((img*masked_noise), EVENT_GRAY_TENSOR_FILL, 1.)
            return res
        else:        
            noise = interp(torch.clamp(torch.rand(img.shape), 0, percent_to_darken), 0, percent_to_darken, 0, 1)
            mask = torch.zeros_like(noise)
            transformed_mask = self._shapeTransforms(mask)
            masked_noise = torch.clamp(transformed_mask+noise, 0, 1)
            masked_noise = transforms.GaussianBlur(kernel_size=self._blur_kernel, sigma=self._blur_sigma)(masked_noise)
            masked_noise = torch.clamp(masked_noise, 0, 1)
            res = torch.clamp((img*masked_noise), EVENT_GRAY_TENSOR_FILL, 1.)
            return res

    def __repr__(self) -> str:
        format_string = self.__class__.__name__
        return format_string