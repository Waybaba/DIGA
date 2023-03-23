import numpy as np

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from . import BaseDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor, Normalize, Compose, RandomHorizontalFlip, RandomCrop, RandomResizedCrop, RandomRotation, RandomAffine, RandomPerspective, RandomApply, RandomChoice, RandomOrder, ColorJitter, Grayscale, GaussianBlur, CenterCrop, FiveCrop, TenCrop, LinearTransformation, Pad, RandomErasing, RandomGrayscale, RandomAffine, RandomPerspective, RandomApply, RandomChoice, RandomOrder, ColorJitter, Grayscale, GaussianBlur, CenterCrop, FiveCrop, TenCrop, LinearTransformation, Pad, RandomErasing, RandomGrayscale
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import random
import os
import torch

from src.datamodules.seg.base_dataset import BaseDataset

class GTA5DataSet(BaseDataset):
	def __init__(self, root, list_path, set='all',
				 max_iters=None, crop_size=(321, 321), mean=(128, 128, 128)):
		super().__init__(root, list_path, set, max_iters, crop_size, None, mean)

		# map to cityscape's ids
		self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
							  19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
							  26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}


	def get_metadata(self, name):
		img_file = self.root / 'images' / name
		label_file = self.root / 'labels' / name
		return img_file, label_file

	def __getitem__(self, index):
		img_file, label_file, name = self.files[index]
		# image = self.get_image(img_file)
		# label = self.get_labels(label_file)
		image = Image.open(img_file).convert("RGB")
		label_copy = Image.open(label_file) 

		# image = self.preprocess(image)
		if (self.set == "train" or self.set == "trainval" or self.set =="all"):
			image, label_copy = self._train_sync_transform(image, label_copy)
		else:
			image, label_copy = self._val_sync_transform(image, label_copy)
		return image, label_copy, np.array(image.shape), name

	def _train_sync_transform(self, img, mask):
		'''
		:param image:  PIL input image
		:param gt_image: PIL input gt_image
		:return:
		'''
		self.random_mirror = True
		self.random_crop = False
		self.resize = True
		self.gaussian_blur = True
		self.crop_size = self.image_size
		if self.random_mirror:
			# random mirror
			if random.random() < 0.5:
				img = img.transpose(Image.FLIP_LEFT_RIGHT)
				if mask: mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
			crop_w, crop_h = self.crop_size

		if self.random_crop:
			# random scale
			base_w , base_h = self.base_size
			w, h = img.size
			assert w >= h
			if (base_w / w) > (base_h / h):
				base_size = base_w 
				short_size = random.randint(int(base_size * 0.5), int(base_size * 2.0))
				ow = short_size
				oh = int(1.0 * h * ow / w)
			else:
				base_size = base_h
				short_size = random.randint(int(base_size * 0.5), int(base_size * 2.0))
				oh = short_size
				ow = int(1.0 * w * oh / h)

			img = img.resize((ow, oh), Image.BICUBIC)
			if mask: mask = mask.resize((ow, oh), Image.NEAREST)
			# pad crop
			if ow < crop_w or oh < crop_h:
				padh = crop_h - oh if oh < crop_h else 0
				padw = crop_w - ow if ow < crop_w else 0
				img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
				if mask: mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
			# random crop crop_size
			w, h = img.size
			x1 = random.randint(0, w - crop_w)
			y1 = random.randint(0, h - crop_h)
			img = img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
			if mask: mask = mask.crop((x1, y1, x1 + crop_w, y1 + crop_h))
		elif self.resize:
			img = img.resize(self.crop_size, Image.BICUBIC)
			if mask: mask = mask.resize(self.crop_size, Image.NEAREST)
		
		if self.gaussian_blur:
			# gaussian blur as in PSP
			if random.random() < 0.5:
				img = img.filter(ImageFilter.GaussianBlur(
					radius=random.random()))
		# final transform
		if mask: 
			img, mask = self._img_transform(img), self._mask_transform(mask)
			return img, mask
		else:
			img = self._img_transform(img)
			return img

	def _val_sync_transform(self, img, mask):
		if self.random_crop:
			crop_w, crop_h = self.crop_size
			w, h = img.size
			if crop_w / w < crop_h / h:
				oh = crop_h
				ow = int(1.0 * w * oh / h)
			else:
				ow = crop_w
				oh = int(1.0 * h * ow / w)
			img = img.resize((ow, oh), Image.BICUBIC)
			mask = mask.resize((ow, oh), Image.NEAREST)
			# center crop
			w, h = img.size
			x1 = int(round((w - crop_w) / 2.))
			y1 = int(round((h - crop_h) / 2.))
			img = img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
			mask = mask.crop((x1, y1, x1 + crop_w, y1 + crop_h))
		elif self.resize:
			img = img.resize(self.crop_size, Image.BICUBIC)
			mask = mask.resize(self.crop_size, Image.NEAREST)

		# final transform
		img, mask = self._img_transform(img), self._mask_transform(mask)
		return img, mask

	def _img_transform(self, image):
		self.numpy_transform = True
		if self.numpy_transform:
			image = np.asarray(image, np.float32)
			image = image[:, :, ::-1]  # change to BGR
			image -= self.mean
			image = image.transpose((2, 0, 1)).copy() # (C x H x W)
			new_image = torch.from_numpy(image)
		else:
			image_transforms = ttransforms.Compose([
				ttransforms.ToTensor(),
				ttransforms.Normalize([.485, .456, .406], [.229, .224, .225]),
			])
			new_image = image_transforms(image)
		return new_image

	def _mask_transform(self, label):
		label = np.asarray(label, np.float32)
		
		# re-assign labels to match the format of Cityscapes
		label_copy = 255 * np.ones(label.shape, dtype=np.float32)
		for k, v in self.id_to_trainid.items():
			label_copy[label == k] = v

		return label_copy

# import 



# if __name__ == "__main__":
# 	dataset = GTA5DataSet('/data/', 'val')
# 	dataset[0]
# 	print(dataset[0])