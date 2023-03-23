from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from pathlib import Path
import os

from src.utils.utils import ratio_lengths_support

from src.datamodules.seg.base_dataset import BaseDataset
from src.datamodules.seg.gta5 import GTA5DataSet as GTA5DataSet_
# from src.datamodules.seg.synthia import SynthiaDataSet as SynthiaDataSet_
from src.datamodules.seg.cityscapes import CityscapesDataSet as CityscapesDataSet_
from src.datamodules.seg.bdd100k import BDD100KDataSet as BDD100KDataSet_
from src.datamodules.seg.mapillary import MapillaryDataSet as MapillaryDataSet_
from src.datamodules.seg.cross_city import CrosscityDataSet as CrosscityDataSet_

import numpy as np
from PIL import Image
from torch.utils import data

import json
import uuid

"""utils"""
def json_load(file_path):
	with open(file_path, 'r') as fp:
		return json.load(fp)

def _load_img(file, size, interpolation, rgb):
	img = Image.open(file)
	if rgb:
		img = img.convert('RGB')
	if size is not None:
		img = img.resize(size, interpolation)
	return np.asarray(img, np.float32)

"""Pytorch Dataset"""
class GTA5DataSet(GTA5DataSet_):
	DATASET_DIR_NAME = 'GTA5'
	CORP_SIZE = (1280,640) # TODO not sure about original size
	def __init__(self, root, set):
		self.data_dir = dataset_dir = root + '/{}/'.format(self.DATASET_DIR_NAME)
		info_dir = dataset_dir + '/advent_list/'
		info_path = info_dir + '/info.json'
		list_path_to_format = info_dir + '/{}.txt'
		super().__init__(
			self.data_dir, list_path_to_format, set,
			crop_size=self.CORP_SIZE if hasattr(self, 'CORP_SIZE') else None,
			mean=np.array((104.00698793,116.66876762,122.67891434))
		)

class CityscapesDataSet(CityscapesDataSet_):
	DATASET_DIR_NAME = 'Cityscapes'
	CORP_SIZE = (1024,512) # TODO not sure about original size
	def __init__(self, root, set):
		self.data_dir = dataset_dir = root + '/{}/'.format(self.DATASET_DIR_NAME)
		info_dir = dataset_dir + '/advent_list/'
		info_path = info_dir + '/info.json'
		list_path_to_format = info_dir + '/{}.txt'
		super().__init__(
			self.data_dir, list_path_to_format, set, 
			info_path=info_path,
			crop_size=self.CORP_SIZE if hasattr(self, 'CORP_SIZE') else None,
			mean=np.array((104.00698793,116.66876762,122.67891434)),
		)

class BDD100KDataSet(BDD100KDataSet_):
	DATASET_DIR_NAME = 'BDD'
	CORP_SIZE = (1280, 720) # seems that the original size is 2048x1024 or others
	def __init__(self, root, set):
		self.data_dir = dataset_dir = root + '/{}/'.format(self.DATASET_DIR_NAME)
		info_dir = dataset_dir + '/advent_list/'
		info_path = info_dir + '/info.json'
		list_path_to_format = info_dir + '/{}.txt'
		super().__init__(
			self.data_dir, set, 
			crop_size=self.CORP_SIZE if hasattr(self, 'CORP_SIZE') else None,
			eval_mode=True,
			mean=np.array((104.00698793,116.66876762,122.67891434)),
		)

class MapillaryDataSet(MapillaryDataSet_):
	DATASET_DIR_NAME = 'Mapillary'
	CORP_SIZE = (1024, 512) # seems that the original size is 2048x1024 or others
	def __init__(self, root, set):
		self.data_dir = dataset_dir = root + '/{}/'.format(self.DATASET_DIR_NAME)
		info_dir = dataset_dir + '/advent_list/'
		info_path = info_dir + '/config.json'
		list_path_to_format = info_dir + '/{}.txt'
		super().__init__(
			self.data_dir, set, "semantic",
			info_path=info_path, 
			crop_size=self.CORP_SIZE if hasattr(self, 'CORP_SIZE') else None,
			mean=np.array((104.00698793,116.66876762,122.67891434)),
		)

class CrosscityDataSet(CrosscityDataSet_):
	DATASET_DIR_NAME = 'NTHU'
	CROP_SIZE = (1024,512) # seems that original size is (2048,1024)
	def __init__(self, root, set, shuffle=False):
		self.data_dir = dataset_dir = root + '/{}/'.format(self.DATASET_DIR_NAME)
		info_dir = dataset_dir + '/advent_list/'
		info_path = info_dir + '/info.json'
		list_path_to_format = info_dir + '/{}.txt'

		super().__init__(
			self.data_dir, list_path_to_format, set, 
			info_path=info_path,
			crop_size= self.CROP_SIZE if hasattr(self, 'CROP_SIZE') else None,
			mean=np.array((104.00698793,116.66876762,122.67891434)),
			shuffle=shuffle,
		)

class IDDDataSet(BaseDataset):
	"""
	The IDD dataset is required to have following folder structure:
	idd/
		leftImg8bit/
					train/city/*_leftImg8bit.png
					test/city/*_leftImg8bit.png
					val/city/*_leftImg8bit.png
		gtFine/
			   train/city/*_gtFine_labelcsTrainIds.png
			   test/city/*_gtFine_labelcsTrainIds.png
			   val/city/*_gtFine_labelcsTrainIds.png
		(there are several appendix of gtFine \
			_gtFine_labelcsTrainIds is the label translated from cityscapes, \
			which is used in this code )
	"""
	DATASET_DIR_NAME = 'IDD'
	CROP_SIZE = (960,540) # seems that original size is 1920x1080
	def __init__(self, root, set, mean=np.array((104.00698793,116.66876762,122.67891434)),):
		self.crop_size = self.CROP_SIZE if hasattr(self, 'CROP_SIZE') else None
		if set!= "small":
			self.set = set
		else:
			self.set = "val"
			set = "val"
			self.sample = True
		self.mean = mean

		self.data_dir = dataset_dir = root + '/{}/'.format(self.DATASET_DIR_NAME)

		# make self.files list (img_file, label_file, name)
		self.img_ids = []
		for city in os.listdir(dataset_dir + '/leftImg8bit/' + set):
			for img_id in os.listdir(dataset_dir + '/leftImg8bit/' + set + '/' + city):
				self.img_ids.append(city+"/"+img_id.split('_')[0])
		self.files = []
		for name in self.img_ids:
			img_file, label_file = self.get_metadata(name)
			self.files.append((img_file, label_file, name))
		if hasattr(self, 'sample'):
			self.files = self.files[:100]
	
	def get_metadata(self, name):
		img_file = self.data_dir + '/leftImg8bit/' + self.set + '/' + name + '_leftImg8bit.png'
		label_file = self.data_dir + '/gtFine/' + self.set + '/' + name + '_gtFine_labelcsTrainIds.png'
		return img_file, label_file
	
	def __len__(self):
		return len(self.files)

	def __getitem__(self, index):
		img_path, label_path, name = self.files[index]
		image = _load_img(img_path, self.crop_size, Image.BICUBIC, True)
		label = _load_img(label_path, self.crop_size, Image.NEAREST, False)
		image = image[:, :, ::-1]  # change to BGR
		if self.mean is not None:
			image -= self.mean
		image = image.transpose((2, 0, 1))
		return image.copy(), label.copy(), np.array(image.shape), name

class MultiInOneDataset(data.Dataset):
	"""A class which combines multiple datasets into one, order is random
	root (str): root directory of the dataset, will be passed to each dataset
	dataset_list (list): a list of dataset objects which will be combined,
		should be inited with the root directory
	crop_size (tuple): crop size of the image, will be passed to each dataset
		if None, the original size will be used
	"""
	def __init__(self, root, dataset_list, crop_size=None):
		self.datasets_list = dataset_list
		self.datasets = []
		# same crop if specified
		if crop_size is not None:
			for dataset in dataset_list:
				self.datasets.append(dataset(root=root, crop_size=crop_size))
		else:
			for dataset in dataset_list:
				self.datasets.append(dataset(root=root))
		# make length and getitem
		self.length = 0
		for dataset in self.datasets:
			self.length += len(dataset)
		self.dataset_indices = np.arange(len(self.datasets))
		self.dataset_indices_cumsum = np.cumsum([len(dataset) for dataset in self.datasets])
		self.dataset_indices_cumsum = np.insert(self.dataset_indices_cumsum, 0, 0)
		self.dataset_indices_cumsum = self.dataset_indices_cumsum[:-1]
	
	def __len__(self):
		return self.length
	
	def __getitem__(self, index):
		dataset_index = np.where(index >= self.dataset_indices_cumsum)[0][-1]
		dataset = self.datasets[dataset_index]
		index = index - self.dataset_indices_cumsum[dataset_index]
		return dataset[index]

"""Lightning Dataset"""
# TODO merge SegmentationDataModule and Sim2RealSegmentationDataModule as one class
class SegmentationDataModule(LightningDataModule):
	"""Example of LightningDataModule for MNIST dataset.
	TODO
	test_list: list of test datset, hydra partial dataset class with split name specified

	A DataModule implements 5 key methods:

		def prepare_data(self):
			# things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
			# download data, pre-process, split, save to disk, etc...
		def setup(self, stage):
			# things to do on every process in DDP
			# load data, set variables, etc...
		def train_dataloader(self):
			# return train dataloader
		def val_dataloader(self):
			# return validation dataloader
		def test_dataloader(self):
			# return test dataloader
		def teardown(self):
			# called on every process in DDP
			# clean up after fit or test

	This allows you to share a full dataset without explaining how to download,
	split, transform and process the data.

	Read the docs:
		https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
	"""

	def __init__(
		self,
		data_dir: str = "data/",
		batch_size: int = 64,
		num_workers: int = 0,
		pin_memory: bool = False,
		train_val_test_split: list = [0.7, 0.1, 0.2],
	):
		super().__init__()

		# this line allows to access init params with 'self.hparams' attribute
		# also ensures init params will be stored in ckpt
		self.save_hyperparameters(logger=False)

		self.setup_info()

		self.data_train: Optional[Dataset] = None
		self.data_val: Optional[Dataset] = None
		self.data_test: Optional[Dataset] = None
	
	def	setup_info(self):
		self.info = {
			"task": "segmentation",
			"num_classes": 19,
			"image_size": (321, 321, ),
			"train_list": ["GTA_train"],
			"val_list": ["GTA_val"],
			"test_list": ["GTA_test"],
			"class_names": ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"],
		}

	def prepare_data(self):
		"""Download data if needed.
		Do not use it to assign state (self.x = y).
		"""
		# TODO check if data exists on disk
		# TODO check if list is empty
		test_list = self.hparams.test_list

	def setup(self, stage: Optional[str] = None):
		"""Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

		This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
		careful not to execute things like random split twice!
		"""
		if not self.data_train and not self.data_val and not self.data_test:
			gta5_dataset = GTA5DataSet(self.hparams.data_dir, set='all')
			gta5_train, gta5_val, gta5_test = random_split(
				dataset=gta5_dataset,
				lengths=ratio_lengths_support(self.hparams.train_val_test_split, len(gta5_dataset)),
				generator=torch.Generator().manual_seed(42),
			)
			self.data_train = gta5_train
			self.data_val = [gta5_val]
			self.data_test = [gta5_test]

	def train_dataloader(self):
		# return empty dataloader
		# get empty subset of self.data_train
		return DataLoader(
			dataset=self.data_train,
			batch_size=self.hparams.batch_size,
			num_workers=self.hparams.num_workers,
			pin_memory=self.hparams.pin_memory,
			shuffle=True,
		)

	def val_dataloader(self):
		if isinstance(self.data_val, list):
			return [
				DataLoader(
				dataset=data_val,
				batch_size=self.hparams.batch_size,
				num_workers=self.hparams.num_workers,
				pin_memory=self.hparams.pin_memory,
				shuffle=False,
			) for data_val in self.data_val
			]
		else:
			return DataLoader(
				dataset=self.data_val,
				batch_size=self.hparams.batch_size,
				num_workers=self.hparams.num_workers,
				pin_memory=self.hparams.pin_memory,
				shuffle=False,
			)

	def test_dataloader(self):
		if isinstance(self.data_test, list):
			return [
				DataLoader(
				dataset=data_test,
				batch_size=self.hparams.batch_size,
				num_workers=self.hparams.num_workers,
				pin_memory=self.hparams.pin_memory,
				shuffle=False,
			) for data_test in self.data_test
			]
		else:
			return DataLoader(
				dataset=self.data_test,
				batch_size=self.hparams.batch_size,
				num_workers=self.hparams.num_workers,
				pin_memory=self.hparams.pin_memory,
				shuffle=False,
			)

	def teardown(self, stage: Optional[str] = None):
		"""Clean up after fit or test."""
		pass

	def state_dict(self):
		"""Extra things to save to checkpoint."""
		return {}

	def load_state_dict(self, state_dict: Dict[str, Any]):
		"""Things to do when loading checkpoint."""
		pass

class Sim2RealSegmentationDataModule(SegmentationDataModule):
	# TODO use this as a base class for all segmentation data modules instead of GTASegmentationDataModule
	def __init__(
		self,
		data_dir: str = "data/",
		batch_size: int = 64,
		num_workers: int = 0,
		pin_memory: bool = False,
		train_list: list = [],
		val_list: list = [],
		test_list: list = [],
	):
		super().__init__(data_dir, batch_size, num_workers, pin_memory)

	def	setup_info(self):
		self.info = {
			"task": "segmentation",
			"num_classes": 19,
			"image_size": (321, 321, ),
			"train_list": self.hparams.train_list,
			"val_list": self.hparams.val_list,
			"test_list": self.hparams.test_list,
			"class_names": ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"],
		}

	def setup(self, stage: Optional[str] = None):
		"""Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

		This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
		careful not to execute things like random split twice!
		"""
		if not self.data_train and not self.data_val and not self.data_test:
			if len(self.hparams.train_list) != 1: raise ValueError("train_list must be a list of length 1")
			self.data_train = self.hparams.train_list[0](root=self.hparams.data_dir)
			self.data_val = [ds(root=self.hparams.data_dir) for ds in self.hparams.val_list]
			self.data_test = [ds(root=self.hparams.data_dir) for ds in self.hparams.test_list]

class PseudoLabelSegmentationDataModule(Sim2RealSegmentationDataModule):
	def __init__(
		self,
		data_dir: str = "data/",
		batch_size: int = 64,
		num_workers: int = 0,
		pin_memory: bool = False,
		train_list: list = [],
		val_list: list = [],
		test_list: list = [],
		pseudo_label: dict = {}
	):
		super().__init__(
			data_dir=data_dir,
			batch_size=batch_size,
			num_workers=num_workers,
			pin_memory=pin_memory,
			train_list=train_list,
			val_list=val_list,
			test_list=test_list
		)
		self.hparams.pseudo_label = pseudo_label

	def prepare_data(self):
		self.pseudo_label_dataset()
	
	def pseudo_label_dataset(self):
		net = self.hparams.pseudo_label.model.net
		# generate random name for save_dir
		self.save_dir = Path(self.hparams.pseudo_label.save_dir_head) / str(uuid.uuid4())
		data_train = self.hparams.train_list[0](root=self.hparams.data_dir)
		train_loader = DataLoader(
			dataset=self.data_train,
			batch_size=self.hparams.batch_size,
			num_workers=self.hparams.num_workers,
			pin_memory=self.hparams.pin_memory,
			shuffle=True,
		)
		# TODO forward and save pseudo label
		pass

if __name__ == "__main__":
	import hydra
	import omegaconf
	import pyrootutils

	root = pyrootutils.setup_root(__file__, pythonpath=True)
	cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "mnist.yaml")
	cfg.data_dir = str(root / "data")
	_ = hydra.utils.instantiate(cfg)
