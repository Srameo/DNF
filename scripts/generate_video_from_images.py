import argparse
import cv2
import os
import glob
from tqdm import tqdm

IMAGES_PATH = ''
SAVE_PATH = ''
FILE_NAME = None
FPS = 24

def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-path', type=str, required=True, help='Path to the image sequences.')
    parser.add_argument('--save-path', type=str, required=True, help='Path to the save path.')
    parser.add_argument('--file-name', type=str, required=True, help='File name.')
    parser.add_argument('--fps', type=int, default=24, help='FPS of the target video.')
    args = parser.parse_args()
    
    global IMAGES_PATH, SAVE_PATH, FPS, FILE_NAME
    IMAGES_PATH = args.images_path
    SAVE_PATH = args.save_path if args.save_path != "" else "runs/CVPR_DEMO/video/videos"
    FPS = args.fps
    FILE_NAME = args.file_name

def main(images_path, save_path, fps, file_name=None, img_postfix='png', vid_postfix='mp4', scale_factor=1.0):
    vid_to_fourcc = {
        'avi': 'DIVX',
        'mp4': 'mp4v'
    }
    
    images = list(sorted(glob.glob(f'{images_path}/*.{img_postfix}')))
    H, W, _ = cv2.imread(images[0], cv2.IMREAD_COLOR).shape
    frame_size = (int(W * scale_factor), int(H * scale_factor))

    os.makedirs(save_path, exist_ok=True)
    file_name = os.path.basename(images_path) if not file_name else file_name
    out = cv2.VideoWriter(f'{save_path}/{file_name}.{vid_postfix}', cv2.VideoWriter_fourcc(*vid_to_fourcc[vid_postfix]), fps, frame_size)

    for filename in tqdm(images):
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        if scale_factor != 1.0:
            img = cv2.resize(img, frame_size, cv2.INTER_NEAREST)
        out.write(img)

    out.release()
    

if __name__ == "__main__":
    parse_options()
    main(IMAGES_PATH, SAVE_PATH, FPS, file_name=FILE_NAME)