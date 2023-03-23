"""ACDC dataset"""
import os
import torch
import numpy as np
import logging
import glob

from PIL import Image
from advent.dataset.base_dataset import BaseDataset
from advent.dataset.seg_data_base import SegmentationDataset
import random



class ACDCDataSet(SegmentationDataset):
    BASE_DIR = 'acdc'
    NUM_CLASS = 19

    def __init__(self, root='/home/hxx/data/davian/segmentation', split='train', mode=None, transform=None, **kwargs):
        super(ACDCDataSet, self).__init__(root, split, mode, transform, **kwargs)
        assert os.path.exists(self.root), "Please put dataset in {SEG_ROOT}/datasets/acdc"
        self.images, self.mask_paths = _get_acdc_pairs(self.root, self.split)
        assert (len(self.images) == len(self.mask_paths))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                              23, 24, 25, 26, 27, 28, 31, 32, 33]
        self._key = np.array([-1, -1, -1, -1, -1, -1,
                              -1, -1, 0, 1, -1, -1,
                              2, 3, 4, -1, -1, -1,
                              5, -1, 6, 7, 8, 9,
                              10, 11, 12, 13, 14, 15,
                              -1, -1, 16, 17, 18])
        self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32')

    def _class_to_index(self, mask):
        values = np.unique(mask)
        for value in values:
            assert (value in self._mapping)
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)
    def _val_sync_transform_resize(self, img, mask):
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size[1])
        y1 = random.randint(0, h - self.crop_size[0])
        img = img.crop((x1, y1, x1 + self.crop_size[1], y1 + self.crop_size[0]))
        mask = mask.crop((x1, y1, x1 + self.crop_size[1], y1 + self.crop_size[0]))

        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.mask_paths[index])
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask, resize=True)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform_resize(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._val_sync_transform_resize(img, mask)
        if self.transform is not None:
            img = self.transform(img)
        img = img*255
        img = img - img.mean()
        return img, mask, np.array(img.shape), os.path.basename(self.images[index])

    def _mask_transform(self, mask):
        target = self._class_to_index(np.array(mask).astype('int32'))
        return torch.LongTensor(np.array(target).astype('int32'))

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0

    @property
    def classes(self):
        """Category names."""
        return ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
                'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
                'truck', 'bus', 'train', 'motorcycle', 'bicycle')


def _get_acdc_pairs(folder, split='train'):
    img_paths = []
    mask_paths = []
    if split == 'test':
        split = 'val'
    img_paths_temp = glob.glob(os.path.join(folder, 'rgb_anon/*/{}/*/*_rgb_anon.png'.format(split)))
    for imgpath in img_paths_temp:
        maskpath = imgpath.replace('/rgb_anon/', '/gt/').replace('rgb_anon.png', 'gt_labelIds.png')
        if os.path.isfile(imgpath) and os.path.isfile(maskpath):
            img_paths.append(imgpath)
            mask_paths.append(maskpath)
        else:
            logging.info('cannot find the mask or image:', imgpath, maskpath)
    logging.info('Found {} images in the folder {}'.format(len(img_paths), folder))
    return img_paths, mask_paths
