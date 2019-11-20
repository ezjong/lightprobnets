from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import random
import numbers
import math
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None


_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
}


# ------------------------------------------------------------------
# Identity transforms
# ------------------------------------------------------------------
class Identity:
    def __init__(self):
        pass

    def __call__(self, image):
        return image


def image_random_gamma(image, min_gamma=0.7, max_gamma=1.5, clip_image=False):
    gamma = np.random.uniform(min_gamma, max_gamma)
    adjusted = torch.pow(image, gamma)
    if clip_image:
        adjusted.clamp_(0.0, 1.0)
    return adjusted


class RandomGamma:
    def __init__(self, min_gamma=0.7, max_gamma=1.5, clip_image=False):
        self._min_gamma = min_gamma
        self._max_gamma = max_gamma
        self._clip_image = clip_image

    def __call__(self, image):
        return image_random_gamma(
            image,
            min_gamma=self._min_gamma,
            max_gamma=self._max_gamma,
            clip_image=self._clip_image)


def image_random_noise(image, min_stddev, max_stddev, clip_image=False):
    stddev = np.random.uniform(min_stddev, max_stddev)
    additive_noise = torch.zeros_like(image).normal_(mean=0.0, std=stddev)
    image += additive_noise
    if clip_image:
        image.clamp_(0.0, 1.0)
    return image


class RandomNoise:
    def __init__(self, min_stddev, max_stddev, clip_image=False):
        self._min_stddev = min_stddev
        self._max_stddev = max_stddev
        self._clip_image = clip_image

    def __call__(self, image):
        return image_random_noise(
            image,
            min_stddev=self._min_stddev,
            max_stddev=self._max_stddev,
            clip_image=self._clip_image)


# ------------------------------------------------------------------
# Allow transformation chains of the type:
#   im1, im2, .... = transform(im1, im2, ...)
# ------------------------------------------------------------------
class TransformChainer:
    def __init__(self, list_of_transforms):
        self._list_of_transforms = list_of_transforms

    def __call__(self, *args):
        list_of_args = list(args)
        for transform in self._list_of_transforms:
            list_of_args = [transform(arg) for arg in list_of_args]
        if len(args) == 1:
            return list_of_args[0]
        else:
            return list_of_args


# ------------------------------------------------------------------
# Allow transformation chains of the type:
#   im1, im2, .... = split( transform( concatenate(im1, im2, ...) ))
# ------------------------------------------------------------------
class ConcatTransformSplitChainer:
    def __init__(self, list_of_transforms, from_numpy=True, to_numpy=False):
        self._chainer = TransformChainer(list_of_transforms)
        self._from_numpy = from_numpy
        self._to_numpy = to_numpy

    def __call__(self, *args):
        num_splits = len(args)

        if self._from_numpy:
            concatenated = np.concatenate(args, axis=0)
        else:
            concatenated = torch.cat(args, dim=1)

        transformed = self._chainer(concatenated)

        if self._to_numpy:
            split = np.split(transformed, indices_or_sections=num_splits, axis=0)
        else:
            split = torch.chunk(transformed, num_splits, dim=1)

        return split


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _get_inverse_affine_matrix(center, angle, translate, scale, shear):
    # Helper method to compute inverse matrix for affine transformation

    # As it is explained in PIL.Image.rotate
    # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RSS is rotation with scale and shear matrix
    #       RSS(a, scale, shear) = [ cos(a)*scale    -sin(a + shear)*scale     0]
    #                              [ sin(a)*scale    cos(a + shear)*scale     0]
    #                              [     0                  0          1]
    # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

    angle = math.radians(angle)
    shear = math.radians(shear)
    scale = 1.0 / scale

    # Inverted rotation matrix with scale and shear
    d = math.cos(angle + shear) * math.cos(angle) + math.sin(angle + shear) * math.sin(angle)
    matrix = [
        math.cos(angle + shear), math.sin(angle + shear), 0,
        -math.sin(angle), math.cos(angle), 0
    ]
    matrix = [scale / d * m for m in matrix]

    # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
    matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (-center[1] - translate[1])
    matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (-center[1] - translate[1])

    # Apply center translation: C * RSS^-1 * C^-1 * T^-1
    matrix[2] += center[0]
    matrix[5] += center[1]
    return matrix


def _affine(img, angle, translate, scale, shear, resample=0, fillcolor=None):
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
        "Argument translate should be a list or tuple of length 2"

    assert scale > 0.0, "Argument scale should be positive"

    output_size = img.size
    center = (img.size[0] * 0.5 + 0.5, img.size[1] * 0.5 + 0.5)
    matrix = _get_inverse_affine_matrix(center, angle, translate, scale, shear)
    return img.transform(output_size, Image.AFFINE, matrix, resample, fillcolor=fillcolor)


class RandomAffine(object):
    def __init__(self, degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        return angle, translations, scale, shear

    def __call__(self, img):
        """
            img (PIL Image): Image to be transformed.

        Returns:
            PIL Image: Affine transformed image.
        """
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img.size)
        return _affine(img, *ret, resample=self.resample, fillcolor=self.fillcolor)

    def __repr__(self):
        s = '{name}(degrees={degrees}'
        if self.translate is not None:
            s += ', translate={translate}'
        if self.scale is not None:
            s += ', scale={scale}'
        if self.shear is not None:
            s += ', shear={shear}'
        if self.resample > 0:
            s += ', resample={resample}'
        if self.fillcolor != 0:
            s += ', fillcolor={fillcolor}'
        s += ')'
        d = dict(self.__dict__)
        d['resample'] = _pil_interpolation_to_str[d['resample']]
        return s.format(name=self.__class__.__name__, **d)
