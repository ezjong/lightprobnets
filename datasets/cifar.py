from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path
import numpy as np
import sys
from PIL import Image
import logging
import os
import os.path
import hashlib
import errno

import torch.utils.data as data
from torchvision import transforms as vision_transforms
from . import transforms

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


# meta data for cifar images and classes
meta = {
    "rgb_mean": (0.4914, 0.4822, 0.4465),
    "rgb_std": (0.2023, 0.1994, 0.2010),
    "classes": ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
}


def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def download_url(url, root, filename, md5):
    from six.moves import urllib

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        logging.info('Using downloaded and verified file: ' + fpath)
    else:
        try:
            logging.info('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath)
        except:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                logging.info('Failed download. Trying https -> http instead.')
                logging.info('Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath)


class _CIFAR10(data.Dataset):
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self,
                 root,
                 train=True,
                 distortion=False,
                 photometric_augmentations=None,
                 affine_augmentations=None,
                 random_flip=False,
                 normalize_colors=False,
                 per_image_std=False,
                 add_noise=False,
                 download=False,
                 crop=None,
                 num_examples=-1):

        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set
        self.per_image_std = per_image_std

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        normalize_colors_transform = transforms.Identity()
        affine_transform = transforms.Identity()
        flip_transform = transforms.Identity()
        noise_transform = transforms.Identity()
        crop_transform = transforms.Identity()

        if crop is not None:
            if train:
                crop_transform = vision_transforms.RandomCrop(crop)
            else:
                crop_transform = vision_transforms.CenterCrop(crop)

        if normalize_colors:
            normalize_colors_transform = vision_transforms.Normalize(
                mean=meta["rgb_mean"], std=meta["rgb_std"])

        self._photometric_transform = transforms.Identity()

        if affine_augmentations is not None:
            affine_transform = transforms.RandomAffine(
                degrees=affine_augmentations["degrees"],
                translate=affine_augmentations["translate"],
                scale=affine_augmentations["scale"],
                shear=affine_augmentations["shear"],
                resample=Image.BICUBIC,
                fillcolor=0)

        if random_flip:
            flip_transform = vision_transforms.RandomHorizontalFlip()

        if add_noise:
            noise_transform = transforms.RandomNoise(min_stddev=0.0, max_stddev=0.02, clip_image=True)

        if photometric_augmentations is not None:
            brightness_max_delta = photometric_augmentations["brightness_max_delta"]
            contrast_max_delta   = photometric_augmentations["contrast_max_delta"]
            saturation_max_delta = photometric_augmentations["saturation_max_delta"]
            hue_max_delta        = photometric_augmentations["hue_max_delta"]
            gamma_min, gamma_max = photometric_augmentations["gamma_min_max"]
            self._photometric_transform = vision_transforms.Compose([
                vision_transforms.ToPILImage(),
                crop_transform,
                vision_transforms.ColorJitter(
                    brightness=brightness_max_delta,
                    contrast=contrast_max_delta,
                    saturation=saturation_max_delta,
                    hue=hue_max_delta),
                affine_transform,
                flip_transform,
                vision_transforms.transforms.ToTensor(),
                transforms.RandomGamma(min_gamma=gamma_min, max_gamma=gamma_max, clip_image=True),
                noise_transform,
                normalize_colors_transform
            ])
        else:
            self._photometric_transform = vision_transforms.Compose([
                vision_transforms.ToPILImage(),
                crop_transform,
                affine_transform,
                flip_transform,
                vision_transforms.transforms.ToTensor(),
                noise_transform,
                normalize_colors_transform
            ])

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        if self.train:
            np_img, target = self.train_data[index], self.train_labels[index]
        else:
            np_img, target = self.test_data[index], self.test_labels[index]

        img = self._photometric_transform(np_img)

        if self.per_image_std:
            m,n = img.size()[1:3]
            mu = img.view(3,-1).mean(dim=1, keepdim=True)
            stddev = img.view(3,-1).std(dim=1, keepdim=True)
            stddev.clamp_(min=(1.0 / np.sqrt(float(m*n))))
            img = (img - mu.view(3,1,1)) / stddev.view(3,1,1)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)


class CifarBase(data.Dataset):
    def __init__(self, cifar):
        super(CifarBase, self).__init__()
        self._cifar = cifar

    def __getitem__(self, index):
        data, target = self._cifar[index]
        example_dict = {
            "input1": data,
            "target1": target,
            "index": index,
            "basename": "img-%05i" % index
        }

        return example_dict

    def __len__(self):
        return len(self._cifar)


# affine_augmentations={ "degrees": [-30, 30],
#                        "translate": [0.3, 0.3],
#                        "scale": [0.8, 1.2],
#                        "shear": [0, 0] },

# photometric_augmentations={ "brightness_max_delta": 0.0,
#                             "contrast_max_delta": 0.0,
#                             "saturation_max_delta": 0.0,
#                             "hue_max_delta": 0.0,
#                             "gamma_min_max": [1.0, 1.0] },

class Cifar10Train(CifarBase):
    def __init__(self,
                 args,
                 root,
                 photometric_augmentations={ "brightness_max_delta": 0.5,
                                             "contrast_max_delta": 0.5,
                                             "saturation_max_delta": 0.5,
                                             "hue_max_delta": 0.0,
                                             "gamma_min_max": [0.9, 1.1] },
                 affine_augmentations={ "degrees": [-5, 5],
                                        "translate": [0.1, 0.1],
                                        "scale": [0.9, 1.1],
                                        "shear": [0, 0] },
                 random_flip=True,
                 add_noise=False,
                 normalize_colors=False,
                 per_image_std=False,
                 crop=None,
                 num_examples=-1):
        d = os.path.dirname(root)
        if not os.path.exists(d):
            os.makedirs(d)
        cifar = _CIFAR10(
            root,
            train=True,
            download=True,
            crop=crop,
            photometric_augmentations=photometric_augmentations,
            affine_augmentations=affine_augmentations,
            random_flip=random_flip,
            add_noise=add_noise,
            normalize_colors=normalize_colors,
            per_image_std=per_image_std,
            num_examples=num_examples)
        super(Cifar10Train, self).__init__(cifar)


class Cifar10Valid(CifarBase):
    def __init__(self,
                 args,
                 root,
                 photometric_augmentations=None,
                 affine_augmentations=None,
                 random_flip=False,
                 add_noise=False,
                 crop=None,
                 normalize_colors=False,
                 per_image_std=False,
                 num_examples=-1):
        d = os.path.dirname(root)
        if not os.path.exists(d):
            os.makedirs(d)
        cifar = _CIFAR10(
            root,
            train=False,
            download=True,
            crop=crop,
            photometric_augmentations=photometric_augmentations,
            affine_augmentations=affine_augmentations,
            random_flip=random_flip,
            add_noise=add_noise,
            per_image_std=per_image_std,
            normalize_colors=normalize_colors,
            num_examples=num_examples)
        super(Cifar10Valid, self).__init__(cifar)
