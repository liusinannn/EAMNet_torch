import os
import sys
import logging
import torch
import numpy as np

from os.path import splitext
from os import listdir
from glob import glob
from torch.utils.data import Dataset
from PIL import Image


class BasicDataset(Dataset):
    def __init__(self, unet_type, imgs_dir, masks_dir1, masks_dir2, masks_dir3, scale=1):
        self.unet_type = unet_type
        self.imgs_dir = imgs_dir
        self.masks_dir1 = masks_dir1
        self.masks_dir2 = masks_dir2
        self.masks_dir3 = masks_dir3
        self.scale = scale

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, unet_type, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'

        pil_img = pil_img.resize((newW, newH))


        img_nd = np.array(pil_img)
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        return img_trans
    #
    # @classmethod
    # def preprocess(cls, pil_img):
    #     img_nd = np.array(pil_img)
    #     if len(img_nd.shape) == 2:
    #         img_nd = np.expand_dims(img_nd, axis=2)
    #
    #     # HWC to CHW
    #     img_trans = img_nd.transpose((2, 0, 1))
    #     # if img_type != 'image':
    #     img_trans = img_trans / 255
    #     return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask1_file = glob(self.masks_dir1 + idx + '.*')
        mask2_file = glob(self.masks_dir2 + idx + '.*')
        mask3_file = glob(self.masks_dir3 + idx + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask1_file) == 1, f'Either no mask or multiple masks found for the ID {idx}: {mask1_file}'
        assert len(mask2_file) == 1, f'Either no mask or multiple masks found for the ID {idx}: {mask2_file}'
        assert len(mask3_file) == 1, f'Either no mask or multiple masks found for the ID {idx}: {mask3_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask1 = Image.open(mask1_file[0])
        mask2 = Image.open(mask2_file[0])
        mask3 = Image.open(mask3_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask1.size, f'Image and mask {idx} should be the same size, but are {img.size} and {mask1.size}'
        assert img.size == mask2.size, f'Image and mask {idx} should be the same size, but are {img.size} and {mask2.size}'
        assert img.size == mask3.size, f'Image and mask {idx} should be the same size, but are {img.size} and {mask3.size}'
        img = self.preprocess(self.unet_type, img, self.scale)
        mask1 = self.preprocess(self.unet_type, mask1, self.scale)
        mask2 = self.preprocess(self.unet_type, mask2, self.scale)
        mask3 = self.preprocess(self.unet_type, mask3, self.scale)

        return {'image': torch.from_numpy(img), 'mask1': torch.from_numpy(mask1), 'mask2': torch.from_numpy(mask2),'mask3': torch.from_numpy(mask3)}
        # return {'image': torch.from_numpy(img), 'mask1': torch.from_numpy(mask1)}
