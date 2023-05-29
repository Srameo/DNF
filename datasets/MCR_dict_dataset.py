#!/usr/bin/env python

import os

import numpy as np
import rawpy
import torch
from torch.utils import data
import time

from utils.registry import DATASET_REGISTRY
from timm.utils import AverageMeter
import imageio
import tqdm

@DATASET_REGISTRY.register()
class MCRDictSet(data.Dataset):

    def __init__(self, data_path, image_list_file, patch_size=None, split='train', load_npy=True, repeat=1,
                 raw_ext='ARW', max_samples=None, max_clip=1.0, min_clip=None, only_00=False,
                 transpose=True, h_flip=True, v_flip=True, rotation=False, ratio=True, **kwargs):
        """
        :param data_path: dataset directory
        :param image_list_file: contains image file names under data_path
        :param patch_size: if None, full images are returned, otherwise patches are returned
        :param split: train or valid
        :param upper: max number of image used for debug
        """
        assert os.path.exists(data_path), "data_path: {} not found.".format(data_path)
        self.data_path = data_path
        image_list_file = os.path.join(data_path, image_list_file)
        assert os.path.exists(image_list_file), "image_list_file: {} not found.".format(image_list_file)
        self.image_list_file = image_list_file
        self.patch_size = patch_size
        self.split = split
        self.load_npy = load_npy
        self.raw_ext = raw_ext
        self.max_clip = max_clip
        self.min_clip = min_clip
        self.transpose = transpose
        self.h_flip = h_flip
        self.v_flip = v_flip
        self.rotation = rotation
        self.ratio = ratio
        self.only_00 = only_00
        self.repeat = repeat

        self.raw_short_read_time = AverageMeter()
        self.raw_short_pack_time = AverageMeter()
        self.raw_short_post_time = AverageMeter()
        self.raw_long_read_time = AverageMeter()
        self.raw_long_pack_time = AverageMeter()
        self.raw_long_post_time = AverageMeter()
        self.npy_long_read_time = AverageMeter()
        self.data_aug_time = AverageMeter()
        self.data_norm_time = AverageMeter()
        self.count = 0

        self.block_size = 2
        self.black_level = 0
        self.white_level = 255

        self.raw_input_path = []
        self.raw_gt_path = []
        self.rgb_gt_path = []
        self.rgb_gt_dict = {}
        self.raw_input_list = []
        self.raw_gt_dict = {}
        with open(self.image_list_file, 'r') as f:
            for i, img_pair in enumerate(f):
                raw_input_path, raw_gt_path, rgb_gt_path = img_pair.strip().split(' ')
                self.raw_input_path.append(os.path.join(self.data_path, raw_input_path))
                self.raw_gt_path.append(os.path.join(self.data_path, raw_gt_path))
                self.rgb_gt_path.append(os.path.join(self.data_path, rgb_gt_path))
                raw_input = imageio.imread(os.path.join(self.data_path, raw_input_path))
                self.raw_input_list.append(raw_input)

                raw_gt = imageio.imread(os.path.join(self.data_path, raw_gt_path))
                raw_gt_name = os.path.basename(raw_gt_path)
                if raw_gt_name not in self.raw_gt_dict:
                    self.raw_gt_dict[raw_gt_name] = raw_gt

                rgb_gt = imageio.imread(os.path.join(self.data_path, rgb_gt_path)).transpose(2, 0, 1)
                rgb_gt_name = os.path.basename(rgb_gt_path)
                if rgb_gt_name not in self.rgb_gt_dict:
                    self.rgb_gt_dict[rgb_gt_name] = rgb_gt

                if max_samples and i == max_samples - 1:  # for debug purpose
                    break

        print("processing: {} images for {}".format(len(self.raw_input_path), self.split))


    def __len__(self):
        return len(self.raw_input_path) * self.repeat

    def print_time(self):
        print('self.raw_short_read_time:', self.raw_short_read_time.avg)
        print('self.raw_short_pack_time:', self.raw_short_pack_time.avg)
        print('self.raw_short_post_time:', self.raw_short_post_time.avg)
        print('self.raw_long_read_time:', self.raw_long_read_time.avg)
        print('self.raw_long_pack_time:', self.raw_long_pack_time.avg)
        print('self.raw_long_post_time:', self.raw_long_post_time.avg)
        print('self.npy_long_read_time:', self.npy_long_read_time.avg)
        print('self.data_aug_time:', self.data_aug_time.avg)
        print('self.data_norm_time:', self.data_norm_time.avg)

    def __getitem__(self, index):
        self.count += 1
        idx = index // self.repeat
        if self.count % 100 == 0 and False:
            self.print_time()
        info = self.raw_input_path[idx]
        img_file = info

        start = time.time()
        noisy_raw = self.raw_input_list[idx]
        if self.patch_size is None:
            # pack raw with patch size is implemented in clip patch for reduce computation
            noisy_raw = self._pack_raw(noisy_raw)
        self.raw_short_read_time.update(time.time() - start)

        lbl_file = self.rgb_gt_path[idx]
        start = time.time()
        clean_rgb = self.rgb_gt_dict[os.path.basename(self.rgb_gt_path[idx])]
        self.raw_long_post_time.update(time.time() - start)

        start = time.time()
        clean_raw = self.raw_gt_dict[os.path.basename(self.raw_gt_path[idx])]
        if self.patch_size is None:
            clean_raw = self._pack_raw(clean_raw)
        self.raw_long_read_time.update(time.time() - start)

        if self.patch_size:
            start = time.time()
            patch_size = self.patch_size
            H, W = clean_rgb.shape[1:3]
            if self.split == 'train':
                if (H - patch_size) // self.block_size > 0:
                    yy = torch.randint(0, (H - patch_size) // self.block_size, (1,))
                else:
                    yy = 0
                if (W - patch_size) // self.block_size > 0:
                    xx = torch.randint(0, (W - patch_size) // self.block_size, (1,))
                else:
                    xx = 0
                # yy, xx = torch.randint(0, (H - patch_size) // self.block_size, (1,)),  torch.randint(0, (W - patch_size) // self.block_size, (1,))
            else:
                yy, xx = (H - patch_size) // self.block_size // 2, (W - patch_size) // self.block_size // 2
            input_patch = self._pack_raw(noisy_raw, yy, xx)
            clean_raw_patch = self._pack_raw(clean_raw, yy, xx)
            gt_patch = clean_rgb[:, yy*self.block_size:yy*self.block_size + patch_size, xx*self.block_size:xx*self.block_size + patch_size]

            if self.h_flip and torch.randint(0, 2, (1,)) == 1 and self.split == 'train':  # random horizontal flip
                input_patch = np.flip(input_patch, axis=2)
                gt_patch = np.flip(gt_patch, axis=2)
                clean_raw_patch = np.flip(clean_raw_patch, axis=2)
            if self.v_flip and torch.randint(0, 2, (1,)) == 1 and self.split == 'train':  # random vertical flip
                input_patch = np.flip(input_patch, axis=1)
                gt_patch = np.flip(gt_patch, axis=1)
                clean_raw_patch = np.flip(clean_raw_patch, axis=1)
            if self.transpose and torch.randint(0, 2, (1,)) == 1 and self.split == 'train':  # random transpose
                input_patch = np.transpose(input_patch, (0, 2, 1))
                gt_patch = np.transpose(gt_patch, (0, 2, 1))
                clean_raw_patch = np.transpose(clean_raw_patch, (0, 2, 1))
            if self.rotation and self.split == 'train':
                raise NotImplementedError('rotation')

            noisy_raw = input_patch.copy()
            clean_rgb = gt_patch.copy()
            clean_raw = clean_raw_patch.copy()
            self.data_aug_time.update(time.time() - start)

        start = time.time()
        noisy_raw = (np.float32(noisy_raw) - self.black_level) / np.float32(self.white_level - self.black_level)  # subtract the black level
        clean_raw = (np.float32(clean_raw) - self.black_level) / np.float32(self.white_level - self.black_level)
        clean_rgb = np.float32(clean_rgb) / np.float32(255)
        self.data_norm_time.update(time.time() - start)

        img_num = int(self.raw_input_path[idx][-23:-20])
        img_expo = int(self.raw_input_path[idx][-8:-4],16)

        if img_num < 500:
            gt_expo = 12287
        else:
            gt_expo = 1023
        ratio = gt_expo / img_expo

        if self.ratio:
            noisy_raw = noisy_raw * ratio
        if self.max_clip is not None:
            noisy_raw = np.minimum(noisy_raw, self.max_clip)
        if self.min_clip is not None:
            noisy_raw = np.maximum(noisy_raw, self.min_clip)

        clean_rgb = clean_rgb.clip(0.0, 1.0)

        noisy_raw = torch.from_numpy(noisy_raw).float()
        clean_rgb = torch.from_numpy(clean_rgb).float()
        clean_raw = torch.from_numpy(clean_raw).float()

        return {
            'noisy_raw': noisy_raw,
            'clean_raw': clean_raw,
            'clean_rgb': clean_rgb,
            'img_file': img_file,
            'lbl_file': lbl_file,
            'img_exposure': img_expo,
            'lbl_exposure': gt_expo,
            'ratio': ratio
        }

    def _pack_raw(self, raw, hh=None, ww=None):
        if self.patch_size is None:
            assert hh is None and ww is None
        # pack Bayer image to 4 channels (RGBG)
        # im = raw.raw_image_visible.astype(np.uint16)

        H, W = raw.shape
        im = np.expand_dims(raw, axis=0)
        if self.patch_size is None:
            out = np.concatenate((im[:, 0:H:2, 0:W:2],
                                  im[:, 0:H:2, 1:W:2],
                                  im[:, 1:H:2, 1:W:2],
                                  im[:, 1:H:2, 0:W:2]), axis=0)
        else:
            h1 = hh * 2
            h2 = hh * 2 + self.patch_size
            w1 = ww * 2
            w2 = ww * 2 + self.patch_size
            out = np.concatenate((im[:, h1:h2:2, w1:w2:2],
                                  im[:, h1:h2:2, w1+1:w2:2],
                                  im[:, h1+1:h2:2, w1+1:w2:2],
                                  im[:, h1+1:h2:2, w1:w2:2]), axis=0)
        return out