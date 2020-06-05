# -*- coding: utf-8 -*-
import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets.data_io import get_transform, read_all_lines
import torchvision.transforms as transforms
import torch
import torch.nn as nn

processed = transforms.Compose([transforms.ToTensor(), ])

# png--tensor--float() / 65535.0
def PngToTensor(image):
    img_tensor = processed(image).float()
    image_tensor = img_tensor / 65535.0
    return image_tensor

class PngLoad(Dataset):
    def __init__(self, datapath, list_filename, training, crop_h, crop_w, channels, resize_num):
        self.datapath = datapath
        self.ori_filenames = self.load_path(list_filename)
        self.training = training
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.channel = channels
        self.resize_num = resize_num

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        ori_images = [x[0] for x in splits]
        return ori_images

    def load_png(self, filename):
        return Image.open(filename)

    def __len__(self):
        return len(self.ori_filenames)

    def __getitem__(self, index):
        ori_img = self.load_png(os.path.join(self.datapath, self.ori_filenames[index]))

        pathname = self.ori_filenames[index]
        if self.training:
            w, h = ori_img.size

            x1 = random.randint(0, w - self.crop_w)
            y1 = random.randint(0, h - self.crop_h)

            # random crop:ori
            img_SR = ori_img.crop((x1, y1, x1 + self.crop_w, y1 + self.crop_h))

            # resize_ratio
            crop_size = (int(self.crop_w / self.resize_num), int(self.crop_h / self.resize_num))
            img_LR = img_SR.resize(crop_size, Image.BICUBIC)

            img_SR = PngToTensor(img_SR)
            img_LR = PngToTensor(img_LR)

            return {"img_lr": img_LR,
                    "img_sr": img_SR}
        else:
            w, h = ori_img.size

            x1 = w - self.crop_w
            y1 = h - self.crop_h

            ori_img = ori_img.crop((x1, y1, x1 + self.crop_w, y1 + self.crop_h))
            # ori_img = processed(ori_img)
            ori_img = PngToTensor(ori_img)
            return {"ori": ori_img,
                    "name": pathname}
