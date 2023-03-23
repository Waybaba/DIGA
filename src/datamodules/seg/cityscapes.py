import numpy as np
import numpy as np
from PIL import Image
from torch.utils import data

import json
import uuid

# from advent.utils import project_root
def json_load(file_path):
	with open(file_path, 'r') as fp:
		return json.load(fp)

def _load_img(file, size, interpolation, rgb):
	img = Image.open(file)
	if rgb:
		img = img.convert('RGB')
	img = img.resize(size, interpolation)
	return np.asarray(img, np.float32)


from src.datamodules.seg.base_dataset import BaseDataset

# DEFAULT_INFO_PATH = project_root / 'advent/dataset/cityscapes_list/info.json'


class CityscapesDataSet(BaseDataset):
    def __init__(self, root, list_path, set='val',
                 max_iters=None,
                 crop_size=(321, 321), mean=(128, 128, 128),
                 load_labels=True,
                 info_path="", labels_size=None):
        super().__init__(root, list_path, set, max_iters, crop_size, labels_size, mean)

        self.load_labels = load_labels
        self.info = json_load(info_path)
        self.class_names = np.array(self.info['label'], dtype=np.str)
        self.mapping = np.array(self.info['label2train'], dtype=np.int)
        self.map_vector = np.zeros((self.mapping.shape[0],), dtype=np.int64)
        for source_label, target_label in self.mapping:
            self.map_vector[source_label] = target_label

    def get_metadata(self, name):
        sub_folder_name = "val" if self.set == "small" else self.set
        img_file = self.root / 'leftImg8bit' / sub_folder_name / name
        label_name = name.replace("leftImg8bit", "gtFine_labelIds")
        label_file = self.root / 'gtFine' / sub_folder_name / label_name
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
