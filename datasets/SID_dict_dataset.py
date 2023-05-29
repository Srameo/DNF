#!/usr/bin/env python

import os

import numpy as np
import rawpy
import torch
from torch.utils import data
import time

from utils.registry import DATASET_REGISTRY
from timm.utils import AverageMeter

class BaseDictSet(data.Dataset):

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

        self.img_info = []
        with open(self.image_list_file, 'r') as f:
            for i, img_pair in enumerate(f):
                img_pair = img_pair.strip()  # ./Sony/short/10003_00_0.04s.ARW ./Sony/long/10003_00_10s.ARW ISO200 F9
                img_file, lbl_file, iso, focus = img_pair.split(' ')
                if self.split == 'test' and self.only_00:
                    if os.path.split(img_file)[-1][5:8] != '_00':
                        continue
                img_exposure = float(os.path.split(img_file)[-1][9:-5]) # 0.04
                lbl_exposure = float(os.path.split(lbl_file)[-1][9:-5]) # 10
                ratio = min(lbl_exposure/img_exposure, 300)
                self.img_info.append({
                    'img': img_file,
                    'lbl': lbl_file,
                    'img_exposure': img_exposure,
                    'lbl_exposure': lbl_exposure,
                    'ratio': np.float32(ratio),
                    'iso': float(iso[3::]),
                    'focus': focus,
                })
                if max_samples and i == max_samples - 1:  # for debug purpose
                    break
        print("processing: {} images for {}".format(len(self.img_info), self.split))


    def __len__(self):
        return len(self.img_info) * self.repeat

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
        if self.count % 100 == 0 and False:
            self.print_time()
        info = self.img_info[index // self.repeat]

        img_file = info['img']
        if not self.load_npy:
            start = time.time()
            raw = rawpy.imread(os.path.join(self.data_path, img_file))
            self.raw_short_read_time.update(time.time() - start)
            start = time.time()
            if self.patch_size is None:
                # pack raw with patch size is implemented in clip patch for reducing computation
                noisy_raw = self._pack_raw(raw)
            self.raw_short_pack_time.update(time.time() - start)
        else:
            start = time.time()
            noisy_raw = np.load(os.path.join(self.data_path, img_file.replace('short', 'short_pack')+'.npy'), allow_pickle=True)
            self.raw_short_read_time.update(time.time() - start)

        lbl_file = info['lbl']
        if self.load_npy:
            start = time.time()
            clean_rgb = np.load(os.path.join(self.data_path, lbl_file.replace('long', 'long_post_int')+'.npy'), allow_pickle=True)
            self.npy_long_read_time.update(time.time() - start)
        else:
            start = time.time()
            lbl_raw = rawpy.imread(os.path.join(self.data_path, lbl_file))
            clean_rgb = lbl_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            clean_rgb = clean_rgb.transpose(2, 0, 1)
            # clean_rgb = clean_rgb / 65535
            self.raw_long_post_time.update(time.time() - start)

        start = time.time()
        if not self.load_npy:
            lbl_raw = rawpy.imread(os.path.join(self.data_path, lbl_file))
            if self.patch_size is None:
                clean_raw = self._pack_raw(lbl_raw)
        else:
            clean_raw = np.load(os.path.join(self.data_path, lbl_file.replace('long', 'long_pack')+'.npy'), allow_pickle=True)
        self.raw_long_read_time.update(time.time() - start)

        if self.patch_size:
            start = time.time()
            patch_size = self.patch_size
            # crop
            H, W = clean_rgb.shape[1:3]
            if self.split == 'train':
                yy, xx = torch.randint(0, (H - patch_size) // self.block_size, (1,)),  torch.randint(0, (W - patch_size) // self.block_size, (1,))
            else:
                yy, xx = (H - patch_size) // self.block_size // 2, (W - patch_size) // self.block_size // 2
            if not self.load_npy:
                input_patch = self._pack_raw(raw, yy, xx)
                clean_raw_patch = self._pack_raw(lbl_raw, yy, xx)
            else:
                input_patch = noisy_raw[:, yy:yy+patch_size//self.block_size, xx:xx+patch_size//self.block_size]
                clean_raw_patch = clean_raw[:, yy:yy+patch_size//self.block_size, xx:xx+patch_size//self.block_size]

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
        clean_rgb = np.float32(clean_rgb) / np.float32(65535)
        self.data_norm_time.update(time.time() - start)

        if self.ratio:
            noisy_raw = noisy_raw * info['ratio']
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
            'img_exposure': info['img_exposure'],
            'lbl_exposure': info['lbl_exposure'],
            'ratio': info['ratio']
        }


@DATASET_REGISTRY.register()
class SonyDictSet(BaseDictSet):
    def __init__(self, data_path, image_list_file, patch_size=None, split='train', load_npy=True, raw_ext='ARW', repeat=1,
                 max_samples=None, max_clip=1.0, min_clip=None, only_00=False,
                 transpose=True, h_flip=True, v_flip=True, rotation=False, ratio=True, **kwargs):
        super(SonyDictSet, self).__init__(data_path, image_list_file, split=split, patch_size=patch_size,
                                   load_npy=load_npy, raw_ext=raw_ext, max_samples=max_samples, max_clip=max_clip, repeat=repeat,
                                   min_clip=min_clip, only_00=only_00, transpose=transpose, h_flip=h_flip, v_flip=v_flip, rotation=rotation, ratio=ratio)
        self.block_size = 2
        self.black_level = 512
        self.white_level = 16383

    def _pack_raw(self, raw, hh=None, ww=None):
        if self.patch_size is None:
            assert hh is None and ww is None
        # pack Bayer image to 4 channels (RGBG)
        im = raw.raw_image_visible.astype(np.uint16)

        H, W = im.shape
        im = np.expand_dims(im, axis=0)
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


@DATASET_REGISTRY.register()
class FujiDictSet(BaseDictSet):
    def __init__(self, data_path, image_list_file, patch_size=None, split='train', load_npy=True, raw_ext='RAF', repeat=1,
                 max_samples=None, max_clip=1.0, min_clip=None, only_00=False,
                 transpose=True, h_flip=True, v_flip=True, rotation=False, ratio=True, **kwargs):
        super(FujiDictSet, self).__init__(data_path, image_list_file, split=split, patch_size=patch_size,
                                   load_npy=load_npy, raw_ext=raw_ext, max_samples=max_samples, max_clip=max_clip, repeat=repeat,
                                   min_clip=min_clip, only_00=only_00, transpose=transpose, h_flip=h_flip, v_flip=v_flip, rotation=rotation, ratio=ratio)
        self.block_size = 3
        self.black_level = 1024
        self.white_level = 16383

    def _pack_raw(self, raw, hh=None, ww=None):
        if self.patch_size is None:
            assert hh is None and ww is None
        # pack XTrans image to 9 channels ()
        im = raw.raw_image_visible.astype(np.uint16)

        H, W = im.shape
        if self.patch_size is None:
            h1 = 0
            h2 = H // 6 * 6
            w1 = 0
            w2 = W // 6 * 6
            out = np.zeros((9, h2 // 3, w2 // 3), dtype=np.uint16)
        else:
            h1 = hh * 3
            h2 = hh * 3 + self.patch_size
            w1 = ww * 3
            w2 = ww * 3 + self.patch_size
            out = np.zeros((9, self.patch_size // 3, self.patch_size // 3), dtype=np.uint16)
        
        # 0 R
        out[0, 0::2, 0::2] = im[h1:h2:6, w1:w2:6]
        out[0, 0::2, 1::2] = im[h1:h2:6, w1+4:w2:6]
        out[0, 1::2, 0::2] = im[h1+3:h2:6, w1+1:w2:6]
        out[0, 1::2, 1::2] = im[h1+3:h2:6, w1+3:w2:6]

        # 1 G
        out[1, 0::2, 0::2] = im[h1:h2:6, w1+2:w2:6]
        out[1, 0::2, 1::2] = im[h1:h2:6, w1+5:w2:6]
        out[1, 1::2, 0::2] = im[h1+3:h2:6, w1+2:w2:6]
        out[1, 1::2, 1::2] = im[h1+3:h2:6, w1+5:w2:6]

        # 1 B
        out[2, 0::2, 0::2] = im[h1:h2:6, w1+1:w2:6]
        out[2, 0::2, 1::2] = im[h1:h2:6, w1+3:w2:6]
        out[2, 1::2, 0::2] = im[h1+3:h2:6, w1:w2:6]
        out[2, 1::2, 1::2] = im[h1+3:h2:6, w1+4:w2:6]

        # 4 R
        out[3, 0::2, 0::2] = im[h1+1:h2:6, w1+2:w2:6]
        out[3, 0::2, 1::2] = im[h1+2:h2:6, w1+5:w2:6]
        out[3, 1::2, 0::2] = im[h1+5:h2:6, w1+2:w2:6]
        out[3, 1::2, 1::2] = im[h1+4:h2:6, w1+5:w2:6]

        # 5 B
        out[4, 0::2, 0::2] = im[h1+2:h2:6, w1+2:w2:6]
        out[4, 0::2, 1::2] = im[h1+1:h2:6, w1+5:w2:6]
        out[4, 1::2, 0::2] = im[h1+4:h2:6, w1+2:w2:6]
        out[4, 1::2, 1::2] = im[h1+5:h2:6, w1+5:w2:6]

        out[5, :, :] = im[h1+1:h2:3, w1:w2:3]
        out[6, :, :] = im[h1+1:h2:3, w1+1:w2:3]
        out[7, :, :] = im[h1+2:h2:3, w1:w2:3]
        out[8, :, :] = im[h1+2:h2:3, w1+1:w2:3]
        return out
