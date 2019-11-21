# from https://github.com/amdegroot/ssd.pytorch
import cv2
import imgaug.augmenters as iaa
import numpy as np
import torch
import torchvision.transforms.functional as TTF
from imgaug.augmentables import Keypoint, KeypointsOnImage
from torchvision import transforms


class ComposeKeyPoints(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, keypoints):
        for t in self.transforms:
            img, keypoints = t(img, keypoints)
        return img, keypoints


class ResizeKeypoints(object):
    def __init__(self, size=224):
        self.size = size

    def __call__(self, image, keypoints):

        if not isinstance(image, np.ndarray):
            image = np.array(image)

        seq = iaa.Sequential([iaa.Resize(self.size)])
        kps = KeypointsOnImage([Keypoint(*kp) for kp in keypoints], shape=image.shape)
        image_aug, kps_aug = seq(image=image, keypoints=kps)
        kps_fixed = np.asarray([[kp.x, kp.y] for kp in kps_aug.keypoints], dtype=np.float32)
        return image_aug, kps_fixed


class ToTensorKeyPoints(object):
    def __call__(self, cvimage, keypoints):
        img_tensor = torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1)
        keypoints_tensor = torch.from_numpy(keypoints.astype(np.float32))
        return img_tensor, keypoints_tensor


class NormalizeKeyPoints(object):
    def __init__(self, mean, std):
        """
        can only be applied after ToTensor, since using the tensor transform of the image and label
        :param mean: double for mean
        :param std: double for std
        """
        super().__init__()
        self.mean = mean
        self.std = std

    def __call__(self, img_tensor, keypoints_tensor):
        """
        apply transforms to the image and label together
        :param img: tensor transform of the PIL grayscale image
        :param label: tensor transform of the 21 landmarks
        :return: normalize image tensor by mean and std , same label
        """

        # move to range of 0-1
        max_v = torch.max(img_tensor)
        img_tensor /= max_v

        # map to (-1,1) range
        keypoints_tensor = ((keypoints_tensor * 2) / img_tensor.shape[-1]) - 1
        
        return TTF.normalize(img_tensor, self.mean, self.std), keypoints_tensor


class To3ChannelsIRKeyPoints(object):
    def __init__(self):
        """
        can only be applied after ToTensor, since using the tensor transform of the image and label
        :param mean: double for mean
        :param std: double for std
        """
        super().__init__()

    def __call__(self, img, keypoints):
        """
        apply transforms to the image and label together
        :param img: tensor transform of the PIL grayscale image
        :param label: tensor transform of the 21 landmarks
        :return: normalize image tensor by mean and std , same label
        """

        # for PIL Image
        try:
            if (img.mode == 'RGB'):
                red = img.getchannel(0)
                to3channel = red.convert('RGB')
            elif (img.mode == 'L'):
                to3channel = img.convert('RGB')
            elif (img.mode == 'BGR'):
                red = img.getchannel(2)
                to3channel = red.convert('RGB')
            elif (img.mode == 'GBR'):
                red = img.getchannel(2)
                to3channel = red.convert('RGB')

            np_img = np.array(to3channel)
            return np_img, keypoints
        except:
            pass

        # for opencv image
        try:
            assert False
            return img, keypoints
        except:
            pass

        return img, keypoints


class RandomMirrorKeyPoints(object):
    def __call__(self, image, keypoints):
        seq = iaa.Sequential([iaa.Fliplr(0.5)])
        kps = KeypointsOnImage([Keypoint(*kp) for kp in keypoints], shape=image.shape)
        image_aug, kps_aug = seq(image=image, keypoints=kps)
        kps_fixed = np.asarray([[kp.x, kp.y] for kp in kps_aug.keypoints], dtype=np.float32)
        return image_aug, kps_fixed


class RandomAffineKeyPoints(object):
    """Rotate the image by random degree.

     Args:
        min_max_angle double: The limits of the degree by which the image can be rotated.
     """

    def __init__(self, min_max_angle):
        super().__init__()
        self.min_max_angle = min_max_angle

    def __call__(self, image, keypoints):
        """
        transform the image and label together
        :param img: PIL grayscale image
        :param label: 21 landmarks
        :return: PIL grayscale image, 21 landmarks
        """

        kps = KeypointsOnImage([Keypoint(*kp) for kp in keypoints], shape=image.shape)

        cval = int(np.random.uniform(20, 230))

        seq = iaa.Sequential([
            iaa.Affine(rotate=self.min_max_angle, cval=cval),
        ])

        # Augment BBs and images.
        image_aug, kps_aug = seq(image=image, keypoints=kps)
        kps_fixed = np.asarray([[kp.x, kp.y] for kp in kps_aug.keypoints], dtype=np.float32)
        return image_aug, kps_fixed
