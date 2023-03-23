import numpy as np

from src.datamodules.seg.base_dataset import BaseDataset
from PIL import Image
import json

def json_load(file_path):
	with open(file_path, 'r') as fp:
		return json.load(fp)

def _load_img(file, size, interpolation, rgb):
	img = Image.open(file)
	if rgb:
		img = img.convert('RGB')
	img = img.resize(size, interpolation)
	return np.asarray(img, np.float32)

# DEFAULT_INFO_PATH = project_root / 'advent/dataset/cityscapes_list/info.json'


class CrosscityDataSet(BaseDataset):
    def __init__(self, root, list_path, set='val',
                 max_iters=None,
                 crop_size=(321, 321), mean=(128, 128, 128),
                 load_labels=True,
                 info_path="", labels_size=None, shuffle=False):
        super().__init__(root, list_path, set, max_iters, crop_size, labels_size, mean, shuffle=shuffle)
        self.load_labels = load_labels
        self.info = json_load(info_path)
        self.class_names = np.array(self.info['label'], dtype=np.str)
        self.mapping = np.array(self.info['label2train'], dtype=np.int)
        self.map_vector = np.zeros((self.mapping.shape[0],), dtype=np.int64)
        for source_label, target_label in self.mapping:
            self.map_vector[source_label] = target_label

    def get_metadata(self, name):
        """
        :param name: image name. e.g. 'Rio/pano_00002_2_180.png'. domain={RIO}, f_name="pano_00002_2_180", ext="png"
        :return:
            img_file: image file path. e.g. 'Rio/Images/{self.set}/pano_00002_2_180.png' -> "{domain}/Images/{self.set}/{f_name}.{ext}"
            label_file: label file path. e.g. 'Rio/Labels/{self.set}/pano_00002_2_180_{eval}.png' -> "{domain}/Labels/{self.set}/{f_name}_eval.png"
        """
        if self.set in ["test", "test_random_order", "Rio", "Tokyo", "Rome", "Taipei"]:
            set_ = "Test"
            domain, f_name = name.split("/")
            f_name, ext = f_name.split(".")
            img_file = self.root / domain / 'Images' / set_ / (f_name + ".{}").format(ext)
            label_file = self.root / domain / 'Labels' / set_ / (f_name + "_eval.png")
        elif self.set == 'all':
            img_file = self.root / 'test/images' / name
            if 'jpg' in name:
                label_name = name.replace(".jpg", "_eval.png")
            else:
                label_name = name.replace(".png", "_eval.png")
            label_file = self.root / 'test/labels' / label_name
        else:
            img_file = self.root / self.set /'Images/Test' / name
            if 'jpg' in name:
                label_name = name.replace(".jpg", "_eval.png")
            else:
                label_name = name.replace(".png", "_eval.png")
            label_file = self.root / self.set /'Labels/Test' / label_name
        return img_file, label_file

    def map_labels(self, input_):
        return self.map_vector[input_.astype(np.int64, copy=False)]

    def __getitem__(self, index):
        img_file, label_file, name = self.files[index]
        label = self.get_labels(label_file)
        label = self.map_labels(label).copy()
        image = self.get_image(img_file)
        image = self.preprocess(image)
        return image.copy(), label, np.array(image.shape), name
