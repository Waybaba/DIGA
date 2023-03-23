from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils import data

class BaseDataset(data.Dataset):
    def __init__(self, root, list_path, set_,
                 max_iters, image_size, labels_size, mean, shuffle=False):
        self.root = Path(root)
        if len(set_.split("_")) == 2 and set_.split("_")[1].isdigit():
            self.set, sample_num = set_.split("_")
            sample_num = int(sample_num)
        else:
            self.set = set_
            sample_num = None
        self.list_path = list_path.format(self.set)
        self.image_size = image_size
        if labels_size is None:
            self.labels_size = self.image_size
        else:
            self.labels_size = labels_size
        self.mean = np.array(mean)
        with open(self.list_path) as f:
            self.img_ids = [i_id.strip() for i_id in f]
        if max_iters is not None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        for name in self.img_ids:
            img_file, label_file = self.get_metadata(name)
            self.files.append((img_file, label_file, name))
        if shuffle:
            np.random.shuffle(self.files)
        # sample
        if sample_num is not None:
            np.random.shuffle(self.files)
            self.files = self.files[:sample_num]

    def get_metadata(self, name):
        raise NotImplementedError

    def __len__(self):
        return len(self.files)

    def preprocess(self, image):
        if self.mean.any():
            image = image[:, :, ::-1]  # change to BGR
            image -= self.mean
        else:
            image = image[:, :, ::-1]  # change to BGR
            image -= image.mean()
        return image.transpose((2, 0, 1))

    def get_image(self, file):
        return _load_img(file, self.image_size, Image.BICUBIC, rgb=True)

    def get_labels(self, file):
        return _load_img(file, self.labels_size, Image.NEAREST, rgb=False)


def _load_img(file, size, interpolation, rgb):
    img = Image.open(file)
    if rgb:
        img = img.convert('RGB')
    if size is not None:
        img = img.resize(size, interpolation)
    return np.asarray(img, np.float32)
