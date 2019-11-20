from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch.utils.data as data
from glob import glob

from torchvision import transforms as vision_transforms

from . import transforms
from . import common

from utils import system


VALIDATE_INDICES = [
    199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210,
    211, 212, 213, 214, 215, 216, 217, 340, 341, 342, 343, 344,
    345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356,
    357, 358, 359, 360, 361, 362, 363, 364, 536, 537, 538, 539,
    540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551,
    552, 553, 554, 555, 556, 557, 558, 559, 560, 659, 660, 661,
    662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673,
    674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685,
    686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697,
    967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978,
    979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990,
    991]


class _Sintel(data.Dataset):
    def __init__(self,
                 args,
                 images_root,
                 flow_root,
                 dstype="train",
                 num_examples=-1):

        self._args = args
        if not os.path.isdir(images_root):
            raise ValueError("Image directory '%s' not found!")
        if flow_root is not None and not os.path.isdir(flow_root):
            raise ValueError("Flow directory '%s' not found!")

        if flow_root is not None:
            all_flo_filenames = sorted(glob(os.path.join(flow_root, "*/*.flo")))
        all_img_filenames = sorted(glob(os.path.join(images_root, "*/*.png")))

        # Remember base for substraction at runtime
        # e.g. subtract_base = "/home/user/.../MPI-Sintel-Complete/training/clean"
        self._substract_base = system.cd_dotdot(images_root)

        # ------------------------------------------------------------------------
        # Get unique basenames
        # ------------------------------------------------------------------------

        # e.g. base_folders = [alley_1", "alley_2", "ambush_2", ...]
        substract_full_base = system.cd_dotdot(all_img_filenames[0])
        base_folders = sorted(list(set([
            os.path.dirname(fn.replace(substract_full_base, ""))[1:] for fn in all_img_filenames
        ])))

        self._image_list = []
        self._flow_list = [] if flow_root is not None else None

        for base_folder in base_folders:
            img_filenames = filter(lambda x: base_folder in x, all_img_filenames)
            if flow_root is not None:
                flo_filenames = filter(lambda x: base_folder in x, all_flo_filenames)

            for i in range(len(img_filenames) - 1):

                im1 = img_filenames[i]
                im2 = img_filenames[i + 1]
                self._image_list += [[ im1, im2 ]]

                if flow_root is not None:
                    flo = flo_filenames[i]
                    self._flow_list += [ flo ]

                # Sanity check
                im1_base_filename = os.path.splitext(os.path.basename(im1))[0]
                im2_base_filename = os.path.splitext(os.path.basename(im2))[0]
                if flow_root is not None:
                    flo_base_filename = os.path.splitext(os.path.basename(flo))[0]
                im1_frame, im1_no = im1_base_filename.split("_")
                im2_frame, im2_no = im2_base_filename.split("_")
                assert(im1_frame == im2_frame)
                assert(int(im1_no) == int(im2_no) - 1)
                if flow_root is not None:
                    flo_frame, flo_no = flo_base_filename.split("_")
                    assert(im1_frame == flo_frame)
                    assert(int(im1_no) == int(flo_no))

        if self._flow_list is not None:
            assert len(self._image_list) == len(self._flow_list)

        # -------------------------------------------------------------
        # Remove invalid validation indices
        # -------------------------------------------------------------
        full_num_examples = len(self._image_list)
        validate_indices = [x for x in VALIDATE_INDICES if x in range(full_num_examples)]

        # ----------------------------------------------------------
        # Construct list of indices for training/validation
        # ----------------------------------------------------------
        list_of_indices = None
        if dstype == "train":
            list_of_indices = [x for x in range(full_num_examples) if x not in validate_indices]
        elif dstype == "valid":
            list_of_indices = validate_indices
        elif dstype == "full":
            list_of_indices = range(full_num_examples)
        else:
            raise ValueError("dstype '%s' unknown!", dstype)

        # ----------------------------------------------------------
        # Restrict dataset indices if num_examples is given
        # ----------------------------------------------------------
        if num_examples > 0:
            restricted_indices = common.deterministic_indices(
                seed=0, k=num_examples, n=len(list_of_indices))
            list_of_indices = [list_of_indices[i] for i in restricted_indices]

        # ----------------------------------------------------------
        # Save list of actual filenames for inputs and flows
        # ----------------------------------------------------------
        self._image_list = [self._image_list[i] for i in list_of_indices]
        if flow_root is not None:
            self._flow_list = [self._flow_list[i] for i in list_of_indices]

        assert len(self._image_list) == len(self._flow_list)

        self._photometric_transform = transforms.ConcatTransformSplitChainer([
            # uint8 -> FloatTensor
            vision_transforms.transforms.ToTensor(),
        ], from_numpy=True, to_numpy=False)

        self._size = len(self._image_list)

    def __getitem__(self, index):
        index = index % self._size

        im1_filename = self._image_list[index][0]
        im2_filename = self._image_list[index][1]

        # read float32 images and flow
        im1_np0 = common.read_image_as_byte(im1_filename)
        im2_np0 = common.read_image_as_byte(im2_filename)

        # possibly apply photometric transformations
        im1, im2 = self._photometric_transform(im1_np0, im2_np0)

        # e.g. "training/clean/alley_1/frame_XXXX"
        basename = os.path.splitext(im1_filename.replace(self._substract_base, "")[1:])[0]

        # import numpy as np
        # from matplotlib import pyplot as plt
        # import numpy as np
        # plt.figure()
        # im1_np = im1.numpy().transpose([1,2,0])
        # im2_np = im2.numpy().transpose([1,2,0])
        # plt.imshow(np.concatenate((im1_np0.astype(np.float32)/255.0, im2_np0.astype(np.float32)/255.0, im1_np, im2_np), 1))
        # plt.show(block=True)

        # example filename
        basename = os.path.basename(im1_filename)[:5]

        example_dict = {
            "input1": im1,
            "input2": im2,
            "index": index,
            "basename": basename
        }

        # convert flow to FloatTensor
        if self._flow_list is not None:
            flo_filename = self._flow_list[index]
            flo_np0 = common.read_flo_as_float32(flo_filename)
            flo = common.numpy2torch(flo_np0)
            example_dict["target1"] = flo

        return example_dict

    def __len__(self):
        return self._size


class SintelTrainingCleanTrain(_Sintel):
    def __init__(self, args, root, num_examples=-1):
        images_root = os.path.join(root, "training", "clean")
        flow_root = os.path.join(root, "training", "flow")
        super(SintelTrainingCleanTrain, self).__init__(
            args,
            images_root=images_root,
            flow_root=flow_root,
            dstype="train",
            num_examples=num_examples)


class SintelTrainingCleanValid(_Sintel):
    def __init__(self, args, root, num_examples=-1):
        images_root = os.path.join(root, "training", "clean")
        flow_root = os.path.join(root, "training", "flow")
        super(SintelTrainingCleanValid, self).__init__(
            args,
            images_root=images_root,
            flow_root=flow_root,
            dstype="valid",
            num_examples=num_examples)


class SintelTrainingCleanFull(_Sintel):
    def __init__(self, args, root, num_examples=-1):
        images_root = os.path.join(root, "training", "clean")
        flow_root = os.path.join(root, "training", "flow")
        super(SintelTrainingCleanFull, self).__init__(
            args,
            images_root=images_root,
            flow_root=flow_root,
            dstype="full",
            num_examples=num_examples)


class SintelTrainingFinalTrain(_Sintel):
    def __init__(self, args, root, num_examples=-1):
        images_root = os.path.join(root, "training", "final")
        flow_root = os.path.join(root, "training", "flow")
        super(SintelTrainingFinalTrain, self).__init__(
            args,
            images_root=images_root,
            flow_root=flow_root,
            dstype="train",
            num_examples=num_examples)


class SintelTrainingFinalValid(_Sintel):
    def __init__(self, args, root, num_examples=-1):
        images_root = os.path.join(root, "training", "final")
        flow_root = os.path.join(root, "training", "flow")
        super(SintelTrainingFinalValid, self).__init__(
            args,
            images_root=images_root,
            flow_root=flow_root,
            dstype="valid",
            num_examples=num_examples)


class SintelTrainingFinalFull(_Sintel):
    def __init__(self, args, root, num_examples=-1):
        images_root = os.path.join(root, "training", "final")
        flow_root = os.path.join(root, "training", "flow")
        super(SintelTrainingFinalFull, self).__init__(
            args,
            images_root=images_root,
            flow_root=flow_root,
            dstype="full",
            num_examples=num_examples)


class SintelTestClean(_Sintel):
    def __init__(self, args, root, num_examples=-1):
        images_root = os.path.join(root, "test", "clean")
        super(SintelTestClean, self).__init__(
            args,
            images_root=images_root,
            dstype="full",
            num_examples=num_examples)


class SintelTestFinal(_Sintel):
    def __init__(self, args, root, num_examples=-1):
        images_root = os.path.join(root, "test", "final")
        super(SintelTestFinal, self).__init__(
            args,
            images_root=images_root,
            dstype="full",
            num_examples=num_examples)
