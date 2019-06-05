from __future__ import division, print_function

import glob
import os
import subprocess
import sys
from shutil import copyfile

import numpy as np
from PIL import Image
from tqdm import tqdm
from vidaug import augmentors as va

VIDAUG_BATCHES = 1


def class_process(src_file, dst_dir_path, class_name):
    dst_class_path = os.path.join(dst_dir_path, class_name)
    if not os.path.exists(dst_class_path):
        os.mkdir(dst_class_path)

    path, file_name_and_ext = os.path.split(src_file)
    name, ext = os.path.splitext(file_name_and_ext)

    path, folder = os.path.split(path)
    dst_directory_path = os.path.join(dst_class_path, f'{folder}__{name}')
    if not os.path.exists(dst_directory_path):
        os.mkdir(dst_directory_path)

    if ext.lower() in ['.jpg', '.jpeg']:
        frame_idx = 1
        copyfile(src_file, f"{dst_directory_path}/image_{frame_idx:05}.jpg")
    elif ext.lower() == '.mov':
        # Using "-loglevel panic" to make ffmpeg quieter
        # cmd = 'ffmpeg -loglevel panic -i \"{}\" -vf "scale=-1:240,fps=16" \"{}/image_%05d.jpg\"'.format(
        cmd = 'ffmpeg -threads 8 -loglevel panic -i \"{}\" -vf scale=-1:240 \"{}/image_%05d.jpg\"'.format(
            src_file, dst_directory_path)
        subprocess.call(cmd, shell=True)
    else:
        print(f"Format of {src_file} is not supported.")
    augmentation(dst_directory_path, class_name)


def augmentation(dst_directory_path, class_name):
    video = []
    for f in glob.iglob(f"{dst_directory_path}/*.jpg"):
        video.append(Image.open(f))

    def sometimes(aug): return va.Sometimes(0.2, aug)
    seq = va.Sequential([
        va.RandomCrop(size=(160, 160)),  # image height = 240
        va.RandomRotate(degrees=10),
        sometimes(va.HorizontalFlip()),
        sometimes(va.GaussianBlur(sigma=1)),
        sometimes(va.InvertColor()),
        sometimes(va.Salt()),
    ])

    for batch_idx in range(VIDAUG_BATCHES):
        video_aug = seq(video)
        for a, v in zip(video_aug, video):
            folder, filename = os.path.split(v.filename)
            folder = folder.replace('__', f'_vidaug_{batch_idx}__')
            if not os.path.exists(folder):
                os.mkdir(folder)
            path = os.path.join(folder, filename)
            a.save(path)


if __name__ == "__main__":
    dir_path = sys.argv[1]
    dst_dir_path = sys.argv[2]

    data_file_pathes = ['/data/data_wedding/labels/train.txt',
                        '/data/data_wedding/labels/val.txt', '/data/data_wedding/labels/test.txt']
    for path in data_file_pathes:
        print(path)
        with open(path, 'r') as fp:
            lines = fp.read().splitlines()
            for l in tqdm(lines):
                tokens = list(filter(None, l.split(',')))
                file_name = tokens[0]
                class_name = tokens[1].strip()
                src_file = os.path.join(dir_path, file_name)
                class_process(src_file, dst_dir_path, class_name)
    print('Done')
