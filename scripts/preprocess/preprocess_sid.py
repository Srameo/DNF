import numpy as np
import rawpy
import os
from glob import glob
from tqdm import tqdm
import argparse
from multiprocessing import Pool
import cv2

def parse_args():
    parser = argparse.ArgumentParser('Preprocess dataset for fast training', add_help=False)

    parser.add_argument('--data-path', type=str, default='./dataset/sid', help='path to dataset')
    parser.add_argument('--camera', type=str, default='Sony', choices=['Sony', 'Fuji'], help='Determine the CFA pattern')
    parser.add_argument('--split', type=str, default='long', choices=['long', 'short'], help='Preprocess long/short split of SID dataset')

    args, unparsed = parser.parse_known_args()
    return args


meta = {
    'Sony': {
        'white_level': 16383,
        'black_level': 512,
        'raw_ext': 'ARW',
        'post_shape': (2848, 4256)
    },
    'Fuji': {
        'white_level': 16383,
        'black_level': 1024,
        'raw_ext': 'RAF',
        'post_shape': (4032, 6030)
    }
}

def pack_raw(im, camera):
    if camera == 'Sony':
        im = np.expand_dims(im, axis=0)
        C, H, W = im.shape
        out = np.concatenate((im[:, 0:H:2, 0:W:2],
                              im[:, 0:H:2, 1:W:2],
                              im[:, 1:H:2, 1:W:2],
                              im[:, 1:H:2, 0:W:2]), axis=0)
    elif camera == 'Fuji':
        img_shape = im.shape
        # orig 4032, 6032
        # crop 4032, 6030
        # pack 1344, 2010

        H = (img_shape[0] // 6) * 6
        W = (img_shape[1] // 6) * 6

        out = np.zeros((9, H // 3, W // 3), dtype=np.uint16)

        # 0 R
        out[0, 0::2, 0::2] = im[0:H:6, 0:W:6]
        out[0, 0::2, 1::2] = im[0:H:6, 4:W:6]
        out[0, 1::2, 0::2] = im[3:H:6, 1:W:6]
        out[0, 1::2, 1::2] = im[3:H:6, 3:W:6]

        # 1 G
        out[1, 0::2, 0::2] = im[0:H:6, 2:W:6]
        out[1, 0::2, 1::2] = im[0:H:6, 5:W:6]
        out[1, 1::2, 0::2] = im[3:H:6, 2:W:6]
        out[1, 1::2, 1::2] = im[3:H:6, 5:W:6]

        # 1 B
        out[2, 0::2, 0::2] = im[0:H:6, 1:W:6]
        out[2, 0::2, 1::2] = im[0:H:6, 3:W:6]
        out[2, 1::2, 0::2] = im[3:H:6, 0:W:6]
        out[2, 1::2, 1::2] = im[3:H:6, 4:W:6]

        # 4 R
        out[3, 0::2, 0::2] = im[1:H:6, 2:W:6]
        out[3, 0::2, 1::2] = im[2:H:6, 5:W:6]
        out[3, 1::2, 0::2] = im[5:H:6, 2:W:6]
        out[3, 1::2, 1::2] = im[4:H:6, 5:W:6]

        # 5 B
        out[4, 0::2, 0::2] = im[2:H:6, 2:W:6]
        out[4, 0::2, 1::2] = im[1:H:6, 5:W:6]
        out[4, 1::2, 0::2] = im[4:H:6, 2:W:6]
        out[4, 1::2, 1::2] = im[5:H:6, 5:W:6]

        out[5, :, :] = im[1:H:3, 0:W:3]
        out[6, :, :] = im[1:H:3, 1:W:3]
        out[7, :, :] = im[2:H:3, 0:W:3]
        out[8, :, :] = im[2:H:3, 1:W:3]
    return out

def preprocess(image_path):
    # print(image_path)
    image_name = os.path.basename(image_path)

    # read raw image
    raw = rawpy.imread(image_path)
    image_visible = raw.raw_image_visible.astype(np.uint16)

    # pack raw image
    pack_image = pack_raw(image_visible, args.camera)

    # save packed image
    save_pack_name = os.path.join(pack_path, image_name)
    np.save(save_pack_name, pack_image, allow_pickle=True)

    # post process raw image using rawpy
    if args.split == 'long':
        post_image = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        # post_image_norm = np.float32(post_image) / 255
        post_image = post_image.transpose(2, 0, 1)
        if post_image.shape[1:] != meta[args.camera]['post_shape']:
            H, W = meta[args.camera]['post_shape']
            post_image = post_image[:, :H, :W]

        save_post_name = os.path.join(post_path, image_name)
        # save_post_jpg_name = os.path.join(post_jpg_path, image_name+'.jpg')

        # save post processed image
        np.save(save_post_name, post_image, allow_pickle=True)
        # save_image(post_image_norm, save_post_jpg_name)

    if args.split == 'long':
        post_image = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        post_image = cv2.cvtColor(post_image, cv2.COLOR_RGB2BGR)
        post_image = (post_image / 65535.0 * 255.0).round().astype(np.uint8)
        if post_image.shape[:2] != meta[args.camera]['post_shape']:
            print(111)
            H, W = meta[args.camera]['post_shape']
            post_image = post_image[:H, :W, :]

        save_post_png_name = os.path.join(post_png_path, image_name + '.png')
        cv2.imwrite(save_post_png_name, post_image)


if __name__ == '__main__':
    args = parse_args()
    print(args)

    data_path = os.path.join(args.data_path, args.camera)
    split_path = os.path.join(data_path, args.split)
    pack_path = os.path.join(data_path, f'{args.split}_pack')
    post_path = os.path.join(data_path, f'{args.split}_post_int')
    post_jpg_path = os.path.join(data_path, f'{args.split}_post_jpg')
    post_png_path = os.path.join(data_path, f'{args.split}_post_png_16')

    os.makedirs(pack_path, exist_ok=True)
    if args.split == 'long':
        os.makedirs(post_path, exist_ok=True)
        os.makedirs(post_jpg_path, exist_ok=True)
        os.makedirs(post_png_path, exist_ok=True)

    image_list = glob(os.path.join(split_path, f'*.{meta[args.camera]["raw_ext"]}'))
    # image_list = image_list[:10]
    print('number of images:', len(image_list))

    with Pool(8) as pool:
        with tqdm(total=len(image_list)) as t:
            for i, x in enumerate(pool.imap(preprocess, image_list)):
                t.update()
