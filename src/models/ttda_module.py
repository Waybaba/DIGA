from configparser import Interpolation
from curses import doupdate, halfdelay
from genericpath import exists
from typing import Any, List
from pathlib import Path

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, JaccardIndex, Metric
from torchmetrics.classification.accuracy import Accuracy
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import logging
from torch.autograd import Variable
import abc
from torchvision.utils import make_grid, draw_segmentation_masks
from copy import deepcopy
from src.models.components import adaptseg
from torch.utils import model_zoo
import copy
from torchvision import transforms
import wandb
import random
from tqdm import tqdm
import time 

logger = logging.getLogger('lightning')
logging.getLogger('PIL').setLevel(logging.INFO)

"""utils"""

class SegmentationUpsample:
	"""upsample depends on input"""
	def __init__(self, size) -> None:
		self.size = size
	
	def __call__(self, x):
		"""
		For x type:
			if input with float tensor, use bilinear interpolation
			if input with long or bool tensor, transform to float tensor and use bilinear interpolation, then transform back to long or bool tensor
		For x dim:
			if input with 4 dim, do nothing
			if input with 3 or 2 dim, unsqueeze to 4 dim then squeeze to original dim
		"""
		origin_dtype = x.dtype
		# interpolation type
		if origin_dtype == torch.float: interpolation = 'nearest' # 'bilinear'
		elif origin_dtype == torch.long or origin_dtype == torch.bool: interpolation = 'nearest'
		else: raise ValueError('input dtype must be float, long or bool')
		# to float tensor
		if origin_dtype == torch.long or origin_dtype == torch.bool: x = x.float()
		# start interpolation
		if x.dim() == 4:
			x = F.interpolate(x, size=self.size, mode=interpolation)
		elif x.dim() == 3:
			x = F.interpolate(x.unsqueeze(0), size=self.size, mode=interpolation).squeeze(0)
		elif x.dim() == 2:
			x = F.interpolate(x.unsqueeze(0).unsqueeze(0), size=self.size, mode=interpolation).squeeze(0).squeeze(0)
		else:
			raise ValueError(f'Input tensor dimension {x.dim()} is not supported')
		# to origin dtype
		x = x.type(origin_dtype)
		return x
"""pytorch model"""
class SegmentationLogger:
	"""TODO check no grad"""
	IMAGE_SIZE = (100, 100)

	@torch.no_grad()
	def __init__(self, model, data, targets, preds, loss, acc, dataset_info={}, plot_limit=5):
		batch_size = targets.shape[0]
		self.model = model
		self.data = data[:plot_limit].cpu()
		self.targets = targets[:plot_limit].cpu()
		self.preds = preds[:plot_limit].cpu()
		self.loss = loss
		self.acc = acc
		self.dataset_info = dataset_info
		self.plot_limit = min(plot_limit, batch_size)
	
	def origin_img(self):
		imgs = [self.data[i] for i in range(self.data.shape[0])]
		return make_grid(imgs, nrow=self.plot_limit, normalize=True)
	
	def img_with_target_seg(self):
		return self._img_with_seg(self.targets)

	def img_with_pred_seg(self):
		return self._img_with_seg(self.preds)

	def img_with_correct_mask(self):
		"""
		mark the correct prediction with green, wrong with red
		"""
		imgs = self.data 
		imgs = (imgs * 255).type(torch.uint8)
		preds = self.preds.argmax(dim=1)
		correct_mask = (self.targets == preds)
		res = [(draw_segmentation_masks(imgs[i], correct_mask[i], alpha=0.5).type(torch.float32) / 255) for i in range(self.plot_limit)]
		res = make_grid(res, nrow=self.plot_limit, normalize=True)
		return res

	def _img_with_seg(self, seg):
		"""
		seg: [batch, 1, H, W] or [batch, H, W]
		"""
		imgs = [self.data[i] for i in range(self.data.shape[0])]
		segs = [seg[i] for i in range(seg.shape[0])]
		return make_grid([self._add_seg(imgs[i], segs[i]) for i in range(self.plot_limit)], nrow=self.plot_limit, normalize=True)

	@torch.no_grad()
	def _add_seg(self, img, seg):
		"""visulize image with segmentation mask with draw_segmentation_masks
		img: [3, H, W]
		seg: [num_classes, H, W] or [H, W]
		"""
		# normalize image to [0, 1] then convert to int8 tensor
		num_classes = self.dataset_info['num_classes']
		img = (img - img.min()) / (img.max() - img.min())
		img = (img * 255).type(torch.uint8)
		if seg.dim() == 3:
			seg = seg.argmax(dim=0)
		# if seg is [H, W], turn to one_hot
		if len(seg.shape) == 2:
			seg[seg == 255] = num_classes
			seg = F.one_hot(seg, num_classes+1)
			seg = seg.permute(2, 0, 1)
		# convert to bool mask tensor
		seg = seg.type(torch.bool)
		res = draw_segmentation_masks(img, seg, alpha=0.5)
		# turn int to float
		res = res.type(torch.float32) / 255
		return res

	def all_wrap(self):
		res = [
			self.origin_img(),
			self.img_with_target_seg(),
			self.img_with_pred_seg(),
			self.img_with_correct_mask()
		]
		# TODO split wrapper from each funciton, the original output is a list of tensor, but the wrapper output is a tensor
		return make_grid(res, nrow=1, normalize=True)
 
class SegmentationMetric(Metric):
	""" Metric for segmentation.
	"""
	def __init__(self, num_classes: int, ignore_index: int = 255):
		super().__init__()
		self.num_classes = num_classes
		self.ignore_index = ignore_index
		self.add_state("hist", default=torch.zeros((self.num_classes, self.num_classes)), dist_reduce_fx="sum")
		# store last batch hist
		self.add_state("last_hist", default=torch.zeros((self.num_classes, self.num_classes)), dist_reduce_fx="sum")
		
	def update(self, preds: torch.Tensor, target: torch.Tensor):
		""" Receives the output of the model and the target.
		"""
		with torch.no_grad():
			batch_size = target.shape[0]
			preds, target = self._input_format(preds, target)
			assert preds.shape == target.shape
			self.last_hist = self._fast_hist(preds.flatten(), target.flatten()) * batch_size
			self.hist += self.last_hist

	def _fast_hist(self, preds: torch.Tensor, target: torch.Tensor):
		"""Compute the histogram.
		"""
		k = (target >= 0) & (target < self.num_classes)
		hist = torch.bincount(self.num_classes * target[k].int() + preds[k], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
		return hist
	
	def _input_format(self, preds: torch.Tensor, target: torch.Tensor):
		"""Convert the input to the correct format.
		"""
		if preds.dim() == 4:
			preds = preds.argmax(dim=1)
		assert preds.dim() == 3
		assert preds.shape == target.shape
		return preds, target

	def _per_class_iou(self, hist: torch.Tensor):
		""" Compute the per class IoU.
		"""
		ious = torch.diag(hist) / (hist.sum(dim=1) + hist.sum(dim=0) - torch.diag(hist))
		return ious

	def compute(self):
		""" Compute the metric."""
		ious = self._per_class_iou(self.hist)
		mean_iou = torch.nanmean(ious) * 100.0
		return mean_iou # TODO add other metrics
	
	def compute_iou(self, type="default"):
		""" Compute the metric."""
		ious = self._per_class_iou(self.hist)
		if type == "default":
			mean_iou = torch.nanmean(ious) * 100.0
		elif type == "16":
			# ignore class 9, 14, 16
			ious = ious[[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17]]
			mean_iou = torch.nanmean(ious) * 100.0
		elif type == "13":
			# ignore class 3, 4, 5, 9, 14, 16
			ious = ious[[0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 15, 17]]
			mean_iou = torch.nanmean(ious) * 100.0
		else: 
			raise NotImplementedError
		return mean_iou # TODO add other metrics

	def compute_class_iou(self):
		""" Compute the per class IoU."""
		ious = self._per_class_iou(self.hist) * 100.0
		return ious
	
	def compute_confusion_matrix(self):
		""" Compute the confusion matrix."""
		return self.hist

affine_par = True

def trans_e_margin(e_margin, class_num):
	# check e margin in 0,1
	if not 0.0 <= e_margin <= 1.0: raise ValueError("e_margin should be 0.-1.")
	# cal min entropy for class_num
	e_max = - class_num * np.log(1/class_num) * (1/class_num)
	e_min = - np.log(1)
	return e_min + (e_max - e_min) * e_margin

def entropy_norm_mul(pred_1, pred_2):
	""" merge two prediction by entropy norm multiply.
	Normlize the entropy of two prediction, then multiply them.

	Args:
		pred_1: [B, C, H, W]
		pred_2: [B, C, H, W]
	Returns:
		merged_pred: [B, C, H, W]
	"""
	pass

class CustomBatchNorm2d(nn.BatchNorm2d):
	def __init__(self, num_features=0, eps=1e-5, momentum=0.1,
				 affine=True, track_running_stats=True):
		super().__init__(
			num_features, eps, momentum, affine, track_running_stats)

	def forward(self, input):
		self._check_input_dim(input)

		exponential_average_factor = 0.0

		if self.training and self.track_running_stats:
			if self.num_batches_tracked is not None:
				self.num_batches_tracked += 1
				if self.momentum is None:  # use cumulative moving average
					exponential_average_factor = 1.0 / float(self.num_batches_tracked)
				else:  # use exponential moving average
					exponential_average_factor = self.momentum

		# calculate running estimates
		if self.training:
			mean = input.mean([0, 2, 3])
			# use biased var in train
			var = input.var([0, 2, 3], unbiased=False)
			n = input.numel() / input.size(1)
			with torch.no_grad():
				self.running_mean = exponential_average_factor * mean\
					+ (1 - exponential_average_factor) * self.running_mean
				# update running_var with unbiased var
				self.running_var = exponential_average_factor * var * n / (n - 1)\
					+ (1 - exponential_average_factor) * self.running_var
		else:
			mean = self.running_mean
			var = self.running_var

		input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
		if self.affine:
			input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

		return input
		model.train()
		forward()
		model.eval()
		forward()

	def from_bn(self, bn):
		self.__init__(
			bn.num_features, bn.eps, bn.momentum,
			bn.affine, bn.track_running_stats
		)
		# copy all self.xxx
		self.load_state_dict(bn.state_dict())
		return self

class SIFABatchNorm2d(CustomBatchNorm2d):
	def forward(self, input):
		self._check_input_dim(input)

		exponential_average_factor = 0.0

		if self.training and self.track_running_stats:
			if self.num_batches_tracked is not None:
				# self.num_batches_tracked.add_(1) # ! removed at Sept. 2022
				if self.momentum is None:  # use cumulative moving average
					exponential_average_factor = 1.0 / float(self.num_batches_tracked)
				else:  # use exponential moving average
					exponential_average_factor = self.momentum

		# if batch_size > 1
		half_first = True
		if half_first == True:
			if input.size(0) > 1:
				mean_cur = (input[:1].mean([0, 2, 3]) + input[1:].mean([0, 2, 3])) / 2 # ! note that we should not use batch_size > 1
				var_cur = (input[:1].var([0, 2, 3], unbiased=False) + input[1:].var([0, 2, 3], unbiased=False)) / 2
			else:
				mean_cur = input.mean([0, 2, 3])
				var_cur = input.var([0, 2, 3], unbiased=False)
		else:
			mean_cur = input.mean([0, 2, 3])
			var_cur = input.var([0, 2, 3], unbiased=False)
		# calculate running estimates
		n = input.numel() / input.size(1)
		if self.training:
			mean, var = mean_cur, var_cur
			with torch.no_grad():
				self.running_mean = exponential_average_factor * mean\
					+ (1 - exponential_average_factor) * self.running_mean
				# update running_var with unbiased var
				self.running_var = exponential_average_factor * var * n / (n - 1)\
					+ (1 - exponential_average_factor) * self.running_var
		else:
			mean = self.lambda_ * self.running_mean + (1-self.lambda_) * mean_cur
			var = self.lambda_ * self.running_var + (1-self.lambda_) * var_cur
		# normal train -> update running mean, var. use current mean, var
		# target 
		# eval -> use self.running_mean, self.running_var
		# source + current 
		input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
		if self.affine:
			input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

		return input
	
	def from_bn(self, bn):
		self.__init__(
			bn.num_features, bn.eps, bn.momentum,
			bn.affine, bn.track_running_stats
		)
		# copy all self.xxx
		self.load_state_dict(bn.state_dict())
		self.lambda_ = torch.tensor(0.)
		return self

class SIFABatchNorm2dTrainable(SIFABatchNorm2d):
	def __init__(self, num_features=0, eps=1e-5, momentum=0.1,
				 affine=True, track_running_stats=True):
		super().__init__(num_features, eps, momentum, affine, track_running_stats)

	def from_bn(self, bn):
		self.__init__(
			bn.num_features, bn.eps, bn.momentum,
			bn.affine, bn.track_running_stats
		)
		# copy all self.xxx
		self.load_state_dict(bn.state_dict())
		self.lambda_ = nn.Parameter(torch.tensor(0.5), requires_grad=True)
		return self

class SourceTargetMeanBatchNorm2d(CustomBatchNorm2d):
	"""Combining source and target mean, var for batch normalization

	The source and target mean, var are combined by a linear combination
	source_mean, source_var is tracked as self.source_mean, self.source_var
	target_mean, target_var is tracked as self.running_mean, self.running_var
	so the updating of self.running_mean, self.running_var is the same
	only combination is different
	"""
	def forward(self, input):
		self._check_input_dim(input)

		exponential_average_factor = 0.0

		if self.training and self.track_running_stats:
			if self.num_batches_tracked is not None:
				# self.num_batches_tracked.add_(1) # ! removed at Sept. 2022
				if self.momentum is None:  # use cumulative moving average
					exponential_average_factor = 1.0 / float(self.num_batches_tracked)
				else:  # use exponential moving average
					exponential_average_factor = self.momentum

		# if batch_size > 1
		half_first = True
		if half_first == True:
			if input.size(0) > 1:
				mean_cur = (input[:1].mean([0, 2, 3]) + input[1:].mean([0, 2, 3])) / 2 # ! note that we should not use batch_size > 1
				var_cur = (input[:1].var([0, 2, 3], unbiased=False) + input[1:].var([0, 2, 3], unbiased=False)) / 2
			else:
				mean_cur = input.mean([0, 2, 3])
				var_cur = input.var([0, 2, 3], unbiased=False)
		else:
			mean_cur = input.mean([0, 2, 3])
			var_cur = input.var([0, 2, 3], unbiased=False)
		# calculate running estimates
		n = input.numel() / input.size(1)
		if self.training:
			mean, var = mean_cur, var_cur
			# with torch.no_grad():
			# 	# if this is the first batch of the target domain, copy it as running mean, var
			# 	# else use running average calculation TODO try track number calculation
			# 	if self.adapt_start == False:
			# 		self.running_mean = mean
			# 		self.running_var = var
			# 		self.adapt_start = torch.tensor(True)
			# 	else:
			# 		self.running_mean = exponential_average_factor * mean\
			# 			+ (1 - exponential_average_factor) * self.running_mean
			# 		self.running_var = exponential_average_factor * var * n / (n - 1)\
			# 			+ (1 - exponential_average_factor) * self.running_var
		else:
			# check adapt start
			# assert self.adapt_start == True, "adapt_start should be True"
			# mean = self.lambda_ * self.running_mean + (1-self.lambda_) * mean_cur
			# var = self.lambda_ * self.running_var + (1-self.lambda_) * var_cur
			# linear combination of self.source_mean, self.source_var and self.running_mean, self.running_var
			exponential_average_factor = self.momentum
			mean, var = mean_cur, var_cur
			with torch.no_grad():
				# if this is the first batch of the target domain, copy it as running mean, var
				# else use running average calculation TODO try track number calculation
				if self.adapt_start == False:
					self.running_mean = mean
					self.running_var = var
					self.adapt_start = torch.tensor(True)
				else:
					self.running_mean = exponential_average_factor * mean\
						+ (1 - exponential_average_factor) * self.running_mean
					self.running_var = exponential_average_factor * var * n / (n - 1)\
						+ (1 - exponential_average_factor) * self.running_var
			mean = self.lambda_ * self.source_mean + (1-self.lambda_) * self.running_mean
			var = self.lambda_ * self.source_var + (1-self.lambda_) * self.running_var

		input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
		if self.affine:
			input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

		return input

	def from_bn(self, bn):
		"""copy from a normal batchnorm layer.
		1. copy all self.xxx
		2. create new buffer for source_mean and source_var
		"""
		self.__init__(
			bn.num_features, bn.eps, bn.momentum,
			bn.affine, bn.track_running_stats
		)
		# copy all self.xxx
		self.load_state_dict(bn.state_dict())
		# get factory_kwargs
		factory_kwargs = {'device': bn.running_mean.device, 'dtype': bn.running_mean.dtype}
		# create a new buffer
		self.register_buffer('source_mean', torch.zeros(bn.num_features, **factory_kwargs))
		self.register_buffer('source_var', torch.ones(bn.num_features, **factory_kwargs))
		# set source as current running, set running as 0.0
		self.source_mean.copy_(bn.running_mean)
		self.source_var.copy_(bn.running_var)
		# register a new buffer call adapt_start as a flag indicating whether the adaptation has started
		self.register_buffer('adapt_start', torch.tensor(False))
		return self

class EvalUpdateBatchNorm2d(CustomBatchNorm2d):
	"""Combining source and target mean, var for batch normalization

	The source and target mean, var are combined by a linear combination
	source_mean, source_var is tracked as self.source_mean, self.source_var
	target_mean, target_var is tracked as self.running_mean, self.running_var
	so the updating of self.running_mean, self.running_var is the same
	only combination is different
	"""
	def forward(self, input):
		self._check_input_dim(input)

		exponential_average_factor = 0.0

		if self.training and self.track_running_stats:
			if self.num_batches_tracked is not None:
				# self.num_batches_tracked.add_(1) # ! removed at Sept. 2022
				if self.momentum is None:  # use cumulative moving average
					exponential_average_factor = 1.0 / float(self.num_batches_tracked)
				else:  # use exponential moving average
					exponential_average_factor = self.momentum

		# if batch_size > 1
		half_first = True
		if half_first == True:
			if input.size(0) > 1:
				mean_cur = (input[:1].mean([0, 2, 3]) + input[1:].mean([0, 2, 3])) / 2 # ! note that we should not use batch_size > 1
				var_cur = (input[:1].var([0, 2, 3], unbiased=False) + input[1:].var([0, 2, 3], unbiased=False)) / 2
			else:
				mean_cur = input.mean([0, 2, 3])
				var_cur = input.var([0, 2, 3], unbiased=False)
		else:
			mean_cur = input.mean([0, 2, 3])
			var_cur = input.var([0, 2, 3], unbiased=False)
		# calculate running estimates
		n = input.numel() / input.size(1)
		if self.training:
			# do nothing
			mean, var = mean_cur, var_cur
		else:
			exponential_average_factor = self.momentum
			mean, var = mean_cur, var_cur
			with torch.no_grad():
				self.running_mean = exponential_average_factor * mean\
					+ (1 - exponential_average_factor) * self.running_mean
				self.running_var = exponential_average_factor * var * n / (n - 1)\
					+ (1 - exponential_average_factor) * self.running_var
			# mean = self.lambda_ * self.source_mean + (1-self.lambda_) * self.running_mean
			# var = self.lambda_ * self.source_var + (1-self.lambda_) * self.running_var
			mean = self.running_mean
			var = self.running_var

		input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
		if self.affine:
			input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

		return input

	def from_bn(self, bn):
		"""copy from a normal batchnorm layer.
		1. copy all self.xxx
		2. create new buffer for source_mean and source_var
		"""
		self.__init__(
			bn.num_features, bn.eps, bn.momentum,
			bn.affine, bn.track_running_stats
		)
		# copy all self.xxx
		self.load_state_dict(bn.state_dict())
		# get factory_kwargs
		factory_kwargs = {'device': bn.running_mean.device, 'dtype': bn.running_mean.dtype}
		# create a new buffer
		self.register_buffer('source_mean', torch.zeros(bn.num_features, **factory_kwargs))
		self.register_buffer('source_var', torch.ones(bn.num_features, **factory_kwargs))
		# set source as current running, set running as 0.0
		self.source_mean.copy_(bn.running_mean)
		self.source_var.copy_(bn.running_var)
		# register a new buffer call adapt_start as a flag indicating whether the adaptation has started
		self.register_buffer('adapt_start', torch.tensor(False))
		return self

class SIFAEvalUpdateBatchNorm2d(EvalUpdateBatchNorm2d):
	"""Combining source and target mean, var for batch normalization

	The source and target mean, var are combined by a linear combination
	source_mean, source_var is tracked as self.source_mean, self.source_var
	target_mean, target_var is tracked as self.running_mean, self.running_var
	so the updating of self.running_mean, self.running_var is the same
	only combination is different
	"""
	def forward(self, input):
		self._check_input_dim(input)

		exponential_average_factor = 0.0

		if self.training and self.track_running_stats:
			if self.num_batches_tracked is not None:
				# self.num_batches_tracked.add_(1) # ! removed at Sept. 2022
				if self.momentum is None:  # use cumulative moving average
					exponential_average_factor = 1.0 / float(self.num_batches_tracked)
				else:  # use exponential moving average
					exponential_average_factor = self.momentum

		# if batch_size > 1
		half_first = True
		if half_first == True:
			if input.size(0) > 1:
				mean_cur = (input[:1].mean([0, 2, 3]) + input[1:].mean([0, 2, 3])) / 2 # ! note that we should not use batch_size > 1
				var_cur = (input[:1].var([0, 2, 3], unbiased=False) + input[1:].var([0, 2, 3], unbiased=False)) / 2
			else:
				mean_cur = input.mean([0, 2, 3])
				var_cur = input.var([0, 2, 3], unbiased=False)
		else:
			mean_cur = input.mean([0, 2, 3])
			var_cur = input.var([0, 2, 3], unbiased=False)
		# calculate running estimates
		n = input.numel() / input.size(1)
		if self.training:
			# do nothing
			mean, var = mean_cur, var_cur
		else:
			exponential_average_factor = self.momentum
			mean, var = mean_cur, var_cur
			with torch.no_grad():
				self.running_mean = exponential_average_factor * mean\
					+ (1 - exponential_average_factor) * self.running_mean
				self.running_var = exponential_average_factor * var * n / (n - 1)\
					+ (1 - exponential_average_factor) * self.running_var
			# mean = self.lambda_ * self.source_mean + (1-self.lambda_) * self.running_mean
			# var = self.lambda_ * self.source_var + (1-self.lambda_) * self.running_var
			mean = self.lambda_ * self.running_mean + (1-self.lambda_) * mean_cur
			var = self.lambda_ * self.running_var + (1-self.lambda_) * var_cur

		input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
		if self.affine:
			input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

		return input

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
	"""Entropy of softmax distribution from logits."""
	temprature = 1
	x = x/ temprature
	x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
	x = x.flatten(1).mean(1)
	return x

def cross_entropy_2d(predict, target):
	"""
	Args:
		predict:(n, c, h, w)
		target:(n, h, w)
	"""
	assert not target.requires_grad
	assert predict.dim() == 4
	assert target.dim() == 3
	assert predict.size(0) == target.size(0), f"{predict.size(0)} vs {target.size(0)}"
	assert predict.size(2) == target.size(1), f"{predict.size(2)} vs {target.size(1)}"
	assert predict.size(3) == target.size(2), f"{predict.size(3)} vs {target.size(3)}"
	n, c, h, w = predict.size()
	target_mask = (target >= 0) * (target != 255)
	target = target[target_mask]
	if not target.data.dim():
		return Variable(torch.zeros(1))
	predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
	predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
	loss = F.cross_entropy(predict, target.long(), reduction="mean")
	return loss

def collect_model_params(model, type=None):
	""" Collect parameters from model, if type is not None, only collect parameters with type
	Args:
		model: model to collect parameters
		type: "bn_all" "bn_weight_bias" "bn_lambda" "bn", "classifier", "classifier_last_layer", "all"
	"""
	if type is None: raise ValueError('type should be specified')
	if type == "all": 
		res = collect_all_params(model)
	elif type in ["bn", "bn_all"]:
		res = collect_bn_params(model, set_=["weight", "bias", "lambda_"])
	elif type == "bn_weight_bias":
		res = collect_bn_params(model, set_=["weight", "bias"])
	elif type == "bn_lambda":
		res = collect_bn_params(model, set_=["lambda_"])
	elif type == "classifier":
		res = collect_classifier_params(model)
	elif type == "classifier_last_layer":
		res = collect_classifier_last_layer_params(model)
	else:
		raise ValueError("type should be one of [bn, bn_all, bn_weight_bias, bn_lambda, classifier, classifier_last_layer, all]")
	return res

def collect_all_params(model):
	""" Collect all parameters from model
	Args:
		model: model to collect parameters
	"""
	params = []
	names = []
	for name, param in model.named_parameters():
		params.append(param)
		names.append(name)
	return params, names

def collect_bn_params(model, set_=['weight', 'bias','lambda_']):
	"""Collect the affine scale + shift parameters from batch norms.
	Walk the model's modules and collect all batch normalization parameters.
	Return the parameters and their names.
	Note: other choices of parameterization are possible!
	"""
	params = []
	names = []
	for nm, m in model.named_modules():
		if isinstance(m, nn.BatchNorm2d):
			for np, p in m.named_parameters():
				if np in set_: # TODO lambda_ is manually added
					params.append(p)
					names.append(f"{nm}.{np}")
	return params, names

def collect_bn_all(model):
	return collect_bn_params(model, set=['weight', 'bias', 'running_mean', 'running_var', 'lambda_'])

def collect_bn_lambda_(model):
	return collect_bn_params(model, set=['lambda_'])

def collect_bn_weight_bias(model):
	return collect_bn_params(model, set=['weight', 'bias'])

def collect_classifier_params(model):
	"""Collect the last module in the model.
	"""
	params = []
	names = []
	for nm, m in model.named_modules():
		if m.__class__.__name__ == "Classifier_Module":
			for np, p in m.named_parameters():
				params.append(p)
				names.append(f"{nm}.{np}")
	return params, names

def collect_classifier_last_layer_params(model):
	"""Collect the last module in the model.
	"""
	params = []
	names = []
	for nm, m in model.named_modules():
		if m.__class__.__name__ == "Classifier_Module":
			# collect params of the last layer in m
			for m in reversed(list(m.modules())):
				if isinstance(m, nn.Conv2d):
					for np, p in m.named_parameters():
						params.append(p)
						names.append(f"{nm}.{np}")
				break
	return params, names

def collect_last_layer_in_last_module_params(model):
	"""Collect the last layer in the model.
	"""
	for m in reversed(list(model.modules())):
		if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
			return m
	raise ValueError("Model does not have a last layer")

class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
		super().__init__()
		# change
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
		self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
		for i in self.bn1.parameters():
			i.requires_grad = False
		padding = dilation
		# change
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
							   padding=padding, bias=False, dilation=dilation)
		self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
		for i in self.bn2.parameters():
			i.requires_grad = False
		self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
		for i in self.bn3.parameters():
			i.requires_grad = False
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)
		out = self.conv3(out)
		out = self.bn3(out)
		if self.downsample is not None:
			residual = self.downsample(x)
		out += residual
		out = self.relu(out)

		return out

class ClassifierModule(nn.Module):
	def __init__(self, inplanes, dilation_series, padding_series, num_classes):
		super(ClassifierModule, self).__init__()
		self.conv2d_list = nn.ModuleList()
		for dilation, padding in zip(dilation_series, padding_series):
			self.conv2d_list.append(
				nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding,
						  dilation=dilation, bias=True))

		for m in self.conv2d_list:
			m.weight.data.normal_(0, 0.01)

	def forward(self, x):
		out = self.conv2d_list[0](x)
		for i in range(len(self.conv2d_list) - 1):
			out += self.conv2d_list[i + 1](x)
		return out

class ResNetMulti(nn.Module):
	def __init__(self, block, layers, num_classes, multi_level, output_size=(224, 224)):
		self.multi_level = multi_level
		self.inplanes = 64
		super(ResNetMulti, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
							   bias=False)
		self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
		for i in self.bn1.parameters():
			i.requires_grad = False
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
		if self.multi_level:
			self.layer5 = ClassifierModule(1024, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
		self.layer6 = ClassifierModule(2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				m.weight.data.normal_(0, 0.01)
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
		self.output_size = output_size

	def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
		downsample = None
		if (stride != 1
				or self.inplanes != planes * block.expansion
				or dilation == 2
				or dilation == 4):
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
		for i in downsample._modules['1'].parameters():
			i.requires_grad = False
		layers = []
		layers.append(
			block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes, dilation=dilation))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		if self.multi_level:
			x1 = self.layer5(x)  # produce segmap 1
		else:
			x1 = None # TODO multi level?
		x2 = self.layer4(x)
		x2 = self.layer6(x2)  # produce segmap 2
		inter = nn.Upsample(size=tuple(self.output_size), mode='bilinear',
								align_corners=True)
		x1, x2 = inter(x1), inter(x2)			
		return x1, x2 # TODO check use x2

	def get_1x_lr_params_no_scale(self):
		"""
		This generator returns all the parameters of the net except for
		the last classification layer. Note that for each batchnorm layer,
		requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
		any batchnorm parameter
		"""
		b = []

		b.append(self.conv1)
		b.append(self.bn1)
		b.append(self.layer1)
		b.append(self.layer2)
		b.append(self.layer3)
		b.append(self.layer4)

		for i in range(len(b)):
			for j in b[i].modules():
				jj = 0
				for k in j.parameters():
					jj += 1
					if k.requires_grad:
						yield k

	def get_10x_lr_params(self):
		"""
		This generator returns all the parameters for the last layer of the net,
		which does the classification of pixel into classes
		"""
		b = []
		if self.multi_level:
			b.append(self.layer5.parameters())
		b.append(self.layer6.parameters())

		for j in range(len(b)):
			for i in b[j]:
				yield i

	def optim_parameters(self, lr):
		return [{'params': self.get_1x_lr_params_no_scale(), 'lr': lr},
				{'params': self.get_10x_lr_params(), 'lr': 10 * lr}]

def get_deeplab_v2(num_classes=19, multi_level=True):
	model = ResNetMulti(Bottleneck, [3, 4, 23, 3], num_classes, multi_level) # TODO class num ? = 19?
	return model

class DeepLabv2(ResNetMulti):
	def __init__(self, num_classes=19, multi_level=True, output_size=(321, 321), restore_from=None):
		super().__init__(Bottleneck, [3, 4, 23, 3], num_classes, multi_level, output_size)
		self.num_classes = num_classes
		if restore_from is not None:
			self.restore_from = restore_from
			self.restore()

	def restore(self):
		if hasattr(self, 'restore_from'):
			restore_from = self.restore_from
			saved_state_dict = torch.load(restore_from)
			if 'state_dict' in saved_state_dict.keys():
				saved_state_dict = saved_state_dict['state_dict']
			if "layer" in tuple(saved_state_dict.keys())[-1].split(".")[1]: 
				start = 1
			elif "layer" in tuple(saved_state_dict.keys())[-1].split(".")[0]:
				start = 0
			else:
				raise ValueError("Can not find layer start.")
			new_params = self.state_dict().copy()
			for i in saved_state_dict: # e.g. self.net.layer3.8.bn1.running_mean -> layer3.8.bn1.running_mean
				i_parts = i.split('.')
				if not self.num_classes == 19 or not i_parts[start] == 'layer5':
					new_params['.'.join(i_parts[start:])] = saved_state_dict[i]
			self.load_state_dict(new_params)

class DeepLabv2AdaptSeg(adaptseg.ResNetMulti):
	def __init__(self, num_classes=19, multi_level=True, output_size=(321, 321), restore_from=None):
		super().__init__(adaptseg.Bottleneck, [3, 4, 23, 3], num_classes)
		self.num_classes = num_classes
		if restore_from is not None:
			self.restore_from = restore_from
			self.restore()

	def restore(self):
		if hasattr(self, 'restore_from'):
			# laod pretrained model parameters
			if self.restore_from[:4] == 'http' :
				saved_state_dict = model_zoo.load_url(self.restore_from)
			else:
				saved_state_dict = torch.load(self.restore_from)
			# replace new model parameters with pretrained model parameters
			if 'imagenet' in self.restore_from.split("/")[-1] or "resnet" in self.restore_from.split("/")[-1]:
				self.load_imagenet_pretrained(saved_state_dict)
			elif 'Cityscapes_source_class13' in self.restore_from.split("/")[-1]:
				self.load_13_pretrained(saved_state_dict["state_dict"])
			elif 'cityscapesbest' in self.restore_from.split("/")[-1] or "models/MaxSquareLoss" in self.restore_from:
				self.load_imagenet_pretrained(saved_state_dict["state_dict"])
			elif 'source' in self.restore_from.split("/")[-1]:
				self.load_source_pretrained(saved_state_dict)
			elif 'my' in self.restore_from.split("/")[-1]:
				self.load_imagenet_pretrained(saved_state_dict['state_dict'])
			elif 'baseline' in self.restore_from.split("/")[-1] or 'dl' in self.restore_from.split("/")[-1]:
				self.load_baseline_pretrained(saved_state_dict)
			else:
				raise ValueError("Can not find pretrained model type.")

	def load_imagenet_pretrained(self, saved_state_dict):
		new_params = self.state_dict().copy()
		for i in saved_state_dict:
			# Scale.layer5.conv2d_list.3.weight
			i_parts = i.split('.')
			if not self.num_classes == 19 or not i_parts[1] == 'layer5':
				new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
		self.load_state_dict(new_params)

	def load_13_pretrained(self, saved_state_dict):
		new_params = self.state_dict().copy()
		for i in saved_state_dict:
			# Scale.layer5.conv2d_list.3.weight
			i_parts = i.split('.')
			if i_parts[1] == 'layer6':
				# i = module.layer6.conv2d_list.0.weight # shape = [13, 2048, 3, 3] or [13,]
				# turn 13 into 19
				new_params['.'.join(i_parts[1:])] = self._param_13_to_19(saved_state_dict[i])
			elif not self.num_classes == 19 or not i_parts[1] == 'layer5':
				new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
		self.load_state_dict(new_params)
	
	def load_source_pretrained(self, saved_state_dict):
		model_dict = self.state_dict()
		saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_dict}
		model_dict.update(saved_state_dict)
		self.load_state_dict(saved_state_dict)
		return

	def load_baseline_pretrained(self, saved_state_dict):
		"""
		miss layer 6, the name is layer5
		format: layer5.conv2d_list.3.weight
		"""
		new_params = self.state_dict().copy()
		for i in saved_state_dict:
			i_parts = i.split('.')
			if i_parts[0] == 'layer5':
				new_params['.'.join(['layer6'] + i_parts[1:])] = saved_state_dict[i]
			elif (not i_parts[0] == 'layer5') and "layer" in i_parts[0]:
				new_params['.'.join(i_parts[0:])] = saved_state_dict[i]
			elif i in new_params.keys():
				new_params[i] = saved_state_dict[i]
			else:
				raise ValueError("Can not find pretrained model type.")
		self.load_state_dict(new_params)

	def _param_13_to_19(self, param):
		""" convert 13 class to 19 class
		the class index is 
			19: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
			16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17]
			13: [0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 15, 17]
			fill non exist class with 0
		input: param, shape = [13, 2048, 3, 3] or [13,]
		return: param, shape = [19, 2048, 3, 3] or [19,]

		"""
		res = torch.zeros(19, *param.shape[1:]).to(param.device)
		res[0] = param[0]
		res[1] = param[1]
		res[2] = param[2]
		res[6] = param[3]
		res[7] = param[4]
		res[8] = param[5]
		res[10] = param[6]
		res[11] = param[7]
		res[12] = param[8]
		res[13] = param[9]
		res[15] = param[10]
		res[17] = param[11]
		res[18] = param[12]
		return res

class DeeplabVGG(adaptseg.DeeplabVGG):
	def __init__(self, num_classes=19, multi_level=True, output_size=(321, 321), restore_from=None):
		super().__init__(num_classes)
		tmp = (adaptseg.Bottleneck, [3, 4, 23, 3])
		self.num_classes = num_classes
		if restore_from is not None:
			self.restore_from = restore_from
			self.restore()

	def restore(self):
		if hasattr(self, 'restore_from'):
			# laod pretrained model parameters
			if self.restore_from[:4] == 'http' :
				saved_state_dict = model_zoo.load_url(self.restore_from)
			else:
				saved_state_dict = torch.load(self.restore_from)
			# replace new model parameters with pretrained model parameters
			if 'imagenet' in self.restore_from.split("/")[-1] or "resnet" in self.restore_from.split("/")[-1]:
				self.load_imagenet_pretrained(saved_state_dict)
			elif 'Cityscapes_source_class13' in self.restore_from.split("/")[-1]:
				self.load_13_pretrained(saved_state_dict["state_dict"])
			elif 'cityscapesbest' in self.restore_from.split("/")[-1] or "models/MaxSquareLoss" in self.restore_from:
				self.load_imagenet_pretrained(saved_state_dict["state_dict"])
			elif 'source' in self.restore_from.split("/")[-1]:
				self.load_source_pretrained(saved_state_dict)
			elif 'my' in self.restore_from.split("/")[-1]:
				self.load_imagenet_pretrained(saved_state_dict['state_dict'])
			elif 'baseline' in self.restore_from.split("/")[-1] or 'dl' in self.restore_from.split("/")[-1]:
				self.load_baseline_pretrained(saved_state_dict)
			else:
				raise ValueError("Can not find pretrained model type.")

	def load_imagenet_pretrained(self, saved_state_dict):
		new_params = self.state_dict().copy()
		for i in saved_state_dict:
			# Scale.layer5.conv2d_list.3.weight
			i_parts = i.split('.')
			if not self.num_classes == 19 or not i_parts[1] == 'layer5':
				new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
		self.load_state_dict(new_params)

	def load_13_pretrained(self, saved_state_dict):
		new_params = self.state_dict().copy()
		for i in saved_state_dict:
			# Scale.layer5.conv2d_list.3.weight
			i_parts = i.split('.')
			if i_parts[1] == 'layer6':
				# i = module.layer6.conv2d_list.0.weight # shape = [13, 2048, 3, 3] or [13,]
				# turn 13 into 19
				new_params['.'.join(i_parts[1:])] = self._param_13_to_19(saved_state_dict[i])
			elif not self.num_classes == 19 or not i_parts[1] == 'layer5':
				new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
		self.load_state_dict(new_params)
	
	def load_source_pretrained(self, saved_state_dict):
		model_dict = self.state_dict()
		saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_dict}
		model_dict.update(saved_state_dict)
		self.load_state_dict(saved_state_dict)
		return

	def load_baseline_pretrained(self, saved_state_dict):
		"""
		miss layer 6, the name is layer5
		format: layer5.conv2d_list.3.weight
		"""
		new_params = self.state_dict().copy()
		for i in saved_state_dict:
			i_parts = i.split('.')
			if i_parts[0] == 'layer5':
				new_params['.'.join(['layer6'] + i_parts[1:])] = saved_state_dict[i]
			elif (not i_parts[0] == 'layer5') and "layer" in i_parts[0]:
				new_params['.'.join(i_parts[0:])] = saved_state_dict[i]
			elif i in new_params.keys():
				new_params[i] = saved_state_dict[i]
			else:
				raise ValueError("Can not find pretrained model type.")
		self.load_state_dict(new_params)

	def _param_13_to_19(self, param):
		""" convert 13 class to 19 class
		the class index is 
			19: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
			16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17]
			13: [0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 15, 17]
			fill non exist class with 0
		input: param, shape = [13, 2048, 3, 3] or [13,]
		return: param, shape = [19, 2048, 3, 3] or [19,]

		"""
		res = torch.zeros(19, *param.shape[1:]).to(param.device)
		res[0] = param[0]
		res[1] = param[1]
		res[2] = param[2]
		res[6] = param[3]
		res[7] = param[4]
		res[8] = param[5]
		res[10] = param[6]
		res[11] = param[7]
		res[12] = param[8]
		res[13] = param[9]
		res[15] = param[10]
		res[17] = param[11]
		res[18] = param[12]
		return res

class BottleneckMaxSquare(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, bn_momentum=0.1):
		super().__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
		self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)

		padding = dilation
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
							   padding=padding, bias=False, dilation=dilation)
		self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)

		self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)

		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out

class Classifier_ModuleMaxSquare(nn.Module):
	def __init__(self, inplanes, dilation_series, padding_series, num_classes):
		super().__init__()
		self.conv2d_list = nn.ModuleList()
		for dilation, padding in zip(dilation_series, padding_series):
			self.conv2d_list.append(
				nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

		for m in self.conv2d_list:
			m.weight.data.normal_(0, 0.01)

	def forward(self, x):
		out = self.conv2d_list[0](x)
		for i in range(len(self.conv2d_list) - 1):
			out += self.conv2d_list[i + 1](x)
			return out

class ResNetMultiMaxSquare(nn.Module):
	def __init__(self, block, layers, num_classes):
		self.inplanes = 64
		super().__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
							   bias=False)
		self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
		for i in self.bn1.parameters():
			i.requires_grad = False
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
		self.layer5 = self._make_pred_layer(Classifier_ModuleMaxSquare, 1024, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
		self.layer6 = self._make_pred_layer(Classifier_ModuleMaxSquare, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, 0.01)
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
		layers = []
		layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes, dilation=dilation))

		return nn.Sequential(*layers)

	def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
		return block(inplanes, dilation_series, padding_series, num_classes)

	def forward(self, x):
		input_size = x.size()[2:]
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)
		x = self.layer1(x)
		x = self.layer2(x)

		x = self.layer3(x)
		x1 = self.layer5(x)
		x1 = F.interpolate(x1, size=input_size, mode='bilinear', align_corners=True)

		x2 = self.layer4(x)
		x2 = self.layer6(x2)
		x2 = F.interpolate(x2, size=input_size, mode='bilinear', align_corners=True)

		return x2, x1 # changed!

	def get_1x_lr_params_NOscale(self):
		"""
		This generator returns all the parameters of the net except for
		the last classification layer. Note that for each batchnorm layer,
		requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
		any batchnorm parameter
		"""
		b = []

		b.append(self.conv1)
		b.append(self.bn1)
		b.append(self.layer1)
		b.append(self.layer2)
		b.append(self.layer3)
		b.append(self.layer4)

		for i in range(len(b)):
			for j in b[i].modules():
				jj = 0
				for k in j.parameters():
					jj += 1
					if k.requires_grad:
						yield k

	def get_10x_lr_params(self):
		"""
		This generator returns all the parameters for the last layer of the net,
		which does the classification of pixel into classes
		"""
		b = []
		b.append(self.layer5.parameters())
		b.append(self.layer6.parameters())

		for j in range(len(b)):
			for i in b[j]:
				yield i

	def optim_parameters(self, args):
		return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.lr},
				{'params': self.get_10x_lr_params(), 'lr': 10 * args.lr}]

class DeepLabv2MaxSquare(ResNetMultiMaxSquare):
	def __init__(self, num_classes=19, multi_level=True, output_size=(321, 321), restore_from=None):
		super().__init__(BottleneckMaxSquare, [3, 4, 23, 3], num_classes)
		
		if restore_from is not None:
			self.restore_from = restore_from
			self.restore()

	def restore(self):
		# if hasattr(self, 'restore_from'):
		# 	not_load_head = False
		# 	restore_from = self.restore_from
		# 	saved_state_dict = torch.load(restore_from)
		# 	if 'state_dict' in saved_state_dict.keys():
		# 		saved_state_dict = saved_state_dict['state_dict']
		# 	if "layer" in tuple(saved_state_dict.keys())[-1].split(".")[1]: 
		# 		start = 1
		# 	elif "layer" in tuple(saved_state_dict.keys())[-1].split(".")[0]:
		# 		start = 0
		# 	else:
		# 		raise ValueError("Can not find layer start.")
		# 	new_params = self.state_dict().copy()
		# 	for i in saved_state_dict: # e.g. self.net.layer3.8.bn1.running_mean -> layer3.8.bn1.running_mean
		# 		i_parts = i.split('.')
		# 		if not i_parts[start] == 'layer5' or not not_load_head:
		# 			new_params['.'.join(i_parts[start:])] = saved_state_dict[i]
		# 	self.load_state_dict(new_params)
		
		if hasattr(self, 'restore_from'):
			checkpoint = torch.load(self.restore_from)
			if 'state_dict' in checkpoint:
				self.load_state_dict(checkpoint['state_dict'])
			else:
				self.load_state_dict(checkpoint)

"""Basic Segmentation"""
class SegmentationBasicModule(LightningModule):
	""" Basic module for segmentation tasks.
	Simply use cross entropy loss as default.
	"""

	def __init__(
		self,
		net: torch.nn.Module,
		optimizer: torch.optim.Optimizer,
		dataset_info: dict,
	):
		super().__init__()

		# this line allows to access init params with 'self.hparams' attribute
		# also ensures init params will be stored in ckpt
		self.save_hyperparameters(logger=False, ignore=["net"])

		self.net = net(
			num_classes=self.hparams.dataset_info["num_classes"],
			output_size=self.hparams.dataset_info["image_size"],
		)

		# use separate metric instance for train, val and test step
		# to ensure a proper reduction over the epoch
		self.train_acc = SegmentationMetric(num_classes=self.hparams.dataset_info["num_classes"], ignore_index=255)
		self.val_acc = nn.ModuleList([SegmentationMetric(num_classes=self.hparams.dataset_info["num_classes"], ignore_index=255) for _ in self.hparams.dataset_info["val_list"]])
		self.test_acc = nn.ModuleList([SegmentationMetric(num_classes=self.hparams.dataset_info["num_classes"], ignore_index=255) for _ in self.hparams.dataset_info["test_list"]])

		# for logging best so far validation accuracy
		self.val_acc_best = nn.ModuleList([MaxMetric()for _ in self.hparams.dataset_info["val_list"]])
		self.val_acc_best_mean = MaxMetric()

	def forward(self, x: torch.Tensor):
		outputs = self.net(x)
		if isinstance(outputs, tuple):
			outputs = outputs[-1]
		return outputs

	def on_train_start(self):
		# by default lightning executes validation step sanity checks before training starts,
		# so we need to make sure val_acc_best doesn't store accuracy from these checks
		self.val_acc_best_mean.reset()
		for val_acc_best in self.val_acc_best: val_acc_best.reset()

	def step(self, batch: Any):
		x, y, shape_, name_ = batch
		outputs = self.forward(x)
		target_size = y.shape[-2:]
		outputs = nn.Upsample(size=target_size, mode='bilinear')(outputs)
		loss, info = self.criterion(outputs, y)
		return loss, outputs, y, info

	def training_step(self, batch: Any, batch_idx: int):
		x, y, shape_, name_ = batch
		loss, preds, targets, info = self.step(batch)

		# log train metrics
		acc = self.train_acc(preds, targets)
		self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
		self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
		# if batch_idx % self.hparams.image_log_interval == 0:
		# if batch_idx % 500 == 0:
		# 	with torch.no_grad():
		# 		elogger = SegmentationLogger(self, batch[0], targets, preds, loss, acc, self.hparams.dataset_info)
		# 		for lg in self.loggers: 
		# 			if "wandb" in lg.__module__:
		# 				wandb = lg
		# 				wandb.log_image(key="train/all_wrap", images=[elogger.all_wrap()])
		# we can return here dict with any tensors
		# and then read it in some callback or in `training_epoch_end()` below
		# remember to always return loss from `training_step()` or else backpropagation will fail!
		return {"loss": loss}

	def training_epoch_end(self, outputs: List[Any]):
		# `outputs` is a list of dicts returned from `training_step()`
		self.train_acc.reset()

	def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
		x, y, shape_, name_ = batch
		loss, preds, targets, info = self.step(batch)
		# log val metrics
		acc = self.val_acc[dataloader_idx](preds, targets)
		self.log(f"val/loss", loss, on_step=True, on_epoch=True, prog_bar=False, add_dataloader_idx=True)
		self.log(f"val/acc", acc, on_step=True, on_epoch=False, prog_bar=True, add_dataloader_idx=True)
		# log images
		# if batch_idx == 0:
		# 	with torch.no_grad():
		# 		elogger = SegmentationLogger(self, x, targets, preds, loss, acc, self.hparams.dataset_info)
		# 		for lg in self.loggers: 
		# 			if "wandb" in lg.__module__:
		# 				wandb = lg
		# 				wandb.log_image(key=f"val/all_wrap_{str(dataloader_idx)}", images=[elogger.all_wrap()])
		return {"loss": loss}

	def validation_epoch_end(self, outputs: List[Any]):
		"""
		here, we only hand `mean` and `best`, others are handled in validation_step
		1. (None) epoch acc is already handled by validation_step
		2. calculate `best for now acc` for each in val_list
		3. calculate `mean acc epoch` for each in val_list 
		4. calculate `best for now mean acc epoch` for each in val_list
		"""
		val_accs = [val_acc.compute() for val_acc in self.val_acc]
		# best (seperately)
		for i in range(len(val_accs)): # log accs of each dataset in val list
			self.val_acc_best[i].update(val_accs[i])
			self.log(f"val/acc/dataloaderr_idx_{str(i)}", val_accs[i], on_epoch=True, prog_bar=True)
			self.log(f"val/acc/dataloaderr_idx_{str(i)}_best", self.val_acc_best[i].compute(), on_epoch=True, prog_bar=True)
		
		# mean and best (mean)
		acc_mean = sum(val_accs) / len(val_accs)
		self.log("val/acc/mean", acc_mean, on_step=False, on_epoch=True, prog_bar=True) # log mean
		self.val_acc_best_mean.update(acc_mean)
		self.log("val/acc/mean_best", self.val_acc_best_mean.compute(), on_epoch=True, prog_bar=True) # log best mean
		
		# for model checkpoint (use last one)
		self.log("val/acc", val_accs[-1], on_step=False, on_epoch=True, prog_bar=False) 
		
		# reset
		for i, val_acc in enumerate(self.val_acc):
			val_acc.reset()

	def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
		loss, preds, targets, info = self.step(batch)
		loss, preds, targets = loss.detach(), preds.detach(), targets.detach()
		# log test metrics
		acc = self.test_acc[dataloader_idx](preds, targets)
		# self.log(f"test/loss", loss, on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=True)
		# self.log(f"test/acc", acc, on_step=True, on_epoch=True, prog_bar=True, add_dataloader_idx=True)
		return {"loss": loss}

	def test_epoch_end(self, outputs: List[Any]):
		test_accs = [test_acc.compute() for test_acc in self.test_acc]
		for i in range(len(test_accs)): # log accs of each dataset in val list
			self.log(f"test/acc/dataloaderr_idx_{str(i)}", test_accs[i], on_epoch=True, prog_bar=True)
		acc_mean = sum(test_accs) / len(test_accs)
		self.log("test/acc/mean", acc_mean, on_step=False, on_epoch=True, prog_bar=True)
		for i, test_acc in enumerate(self.test_acc):
			test_acc.reset()

	def configure_optimizers(self):
		"""Choose what optimizers and learning-rate schedulers to use in your optimization.
		Normally you'd need one. But in the case of GANs or similar you might have multiple.

		Examples:
			https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
		"""
		op = self.hparams.optimizer
		return {
			"optimizer": op(
				# params=self.parameters()
				params=self.net.optim_parameters(op.keywords["lr"]), 
			),
		}

	def criterion(self, outputs, targets):
		loss = cross_entropy_2d(outputs, targets)
		return loss.mean(), {}

class SegmentationSingleLevelModule(SegmentationBasicModule):
	pass

class SegmentationMultiLevelModule(SegmentationBasicModule):
	def __init__(
		self,
		net: torch.nn.Module,
		optimizer: torch.optim.Optimizer,
		dataset_info: dict,
		lambda_aux: float = 0.4,
	):
		super().__init__(net, optimizer, dataset_info)
	
	def forward(self, x):
		return self.net(x)
	
	def step(self, batch):
		x, y, shape_, name_ = batch
		preds, preds_aux = self.net(x)
		preds = nn.Upsample(size=y.shape[-2:], mode='bilinear')(preds)
		preds_aux = nn.Upsample(size=y.shape[-2:], mode='bilinear')(preds_aux)
		loss_main, info = self.criterion(preds, y)
		loss_aux, info = self.criterion(preds_aux, y)
		loss = loss_main + self.hparams.lambda_aux * loss_aux
		return loss, preds, y, info

	def configure_optimizers(self):
		"""Choose what optimizers and learning-rate schedulers to use in your optimization.
		Normally you'd need one. But in the case of GANs or similar you might have multiple.

		Examples:
			https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
		"""
		op = self.hparams.optimizer
		op = op(
				# params=self.parameters()
				params=self.net.optim_parameters(op.keywords["lr"]), 
			)
		return {
			"optimizer": op,
			# poly lr scheduler. # new_lr = init_lr * (1 - float(iter) / max_iter) ** power. power = 0.9
			"lr_scheduler": {
				"scheduler": torch.optim.lr_scheduler.LambdaLR(
					op, 
					lambda step: (1 - step / 200000) ** 0.9 # from MaxSquareLoss paper
				),
				"interval": "step",
				"frequency": 1,
			}
		}


"""DIGA"""
class DIGA(LightningModule):
	def __init__(
		self,
		net: torch.nn.Module,
		dataset_info: dict = {},
		cfg: object = None,
	):
		super().__init__()
		self.save_hyperparameters(logger=False, ignore=["net"])
		
		self.net = net(
			num_classes=self.hparams.dataset_info["num_classes"],
			output_size=self.hparams.dataset_info["image_size"],
		)

		# loss function
		self.criterion = torch.nn.CrossEntropyLoss()

		# use separate metric instance for train, val and test step
		# to ensure a proper reduction over the epoch
		self.test_acc = nn.ModuleList([SegmentationMetric(num_classes=self.hparams.dataset_info["num_classes"], ignore_index=255).cpu() for _ in self.hparams.dataset_info["test_list"]])

		# for logging best so far validation accuracy
		self.val_acc_best = MaxMetric()
		
		self.test_step_global = 0

		self.save_hyperparameters(logger=False, ignore=["net"])
		self._replace_bn()
		self._configure_bn_running_stats()

	# lightning callback
	def on_test_start(self):
		# metric to cpu
		for metric in self.test_acc:
			metric.cpu()

	def forward(self, x: torch.Tensor):
		outputs = self.net(x)
		if isinstance(outputs, tuple):
			outputs = outputs[-1]
		return outputs
	
	def step(self, batch: Any):
		x, y, shape_, name_ = batch
		target_size = y.shape[-2:]
		outputs, info = self.forward_and_adapt((x,y))
		loss = torch.tensor(0.0, device=self.device)
		outputs = nn.Upsample(size=target_size, mode='bilinear')(outputs)
		return loss.cpu(), outputs.cpu(), y.cpu(), info
	
	def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
		x, y, shape_, name_ = batch
		loss, preds, targets, info = self.step(batch)
		# log test metrics
		acc = self.test_acc[dataloader_idx](preds, targets)
		self.log(f"test/loss", loss, on_step=True, on_epoch=True, prog_bar=False, add_dataloader_idx=True)
		self.log(f"test/acc", acc, on_step=True, on_epoch=True, prog_bar=True, add_dataloader_idx=True)
		self.test_step_global += 1
		return {"loss": loss}
	
	def forward_and_adapt(self, pair):
		"""Forward and adapt model on batch of data.
		Measure entropy of the model prediction, take gradients, and update params.
		Return: 
		1. model outputs; 
		2. the number of reliable and non-redundant samples; 
		3. the number of reliable samples;
		4. the moving average  probability vector over all previous samples
		"""
		x, y = pair
		# no_grad
		feature, outputs = self.net(x, feat=True)
		to_logs = {}
		outputs_proto, to_logs_ = self.multi_proto_label(
			feature,
			outputs, 
			y, 
			self.hparams.dataset_info.class_names,
			self.hparams.cfg,
		)

		outputs = outputs_proto * self.hparams.cfg.fusion_lambda + outputs.softmax(1) * (1 - self.hparams.cfg.fusion_lambda)
		return outputs.cpu(), to_logs
	
	def test_epoch_end(self, outputs: List[Any]):
		test_accs = [test_acc.compute() for test_acc in self.test_acc]
		acc_mean = sum(test_accs) / len(test_accs)
		self.log("test/acc/mean", acc_mean, on_step=False, on_epoch=True, prog_bar=True)
		for i in range(len(test_accs)): # log accs of each dataset in val list
			self.log(f"test/acc/dataloaderr_idx_{str(i)}", test_accs[i], on_epoch=True, prog_bar=True)
			self.log(f"test/acc/dataloaderr16_idx_{str(i)}", self.test_acc[i].compute_iou(type="16"), on_epoch=True, prog_bar=True)
			self.log(f"test/acc/dataloaderr13_idx_{str(i)}", self.test_acc[i].compute_iou(type="13"), on_epoch=True, prog_bar=True)
			# use wandb logger to log 
			class_iou = self.test_acc[i].compute_class_iou()
			class_names = self.hparams.dataset_info.class_names
			self.wandb.log_table(key=f"test/class_iou/dataloaderr_idx_{str(i)}", columns=list(class_names), data=[class_iou])
		for i, test_acc in enumerate(self.test_acc): # reset
			test_acc.reset()

	def high_confident_proto_label(self, outputs, threshold):
		"""High confident pseudo label.
		For each pixel, output the max prob class if max_prob > bar, else 255
		Args:
			outputs: (B, C, H, W), logits
			bar: float, threshold
			class_balance: bool, if True, use class balance to select the max prob class
			ground_truth: (B, H, W), ground truth label
		Returns:
			pseudo_label: (B, H, W)
		"""
		outputs = outputs.softmax(dim=1)
		pseudo_label = outputs.argmax(dim=1)
		pseudo_label[outputs.max(dim=1)[0] < threshold] = 255
		return pseudo_label
	
	def multi_proto_label(self, feature, outputs, y, class_names, cfg):
		""" Calculate label for each pixel based on multiple prototypes of each class
		Args:
			feature: (B, CH, H, W), feature map
			outputs: (B, C, H, W), logits
			y: (B, H, W), ground truth
			class_names: list of class names
			cfg: dict, config for multi_proto_label
				strategy: str. values can be "mean_and_instance", "two_proto_per_class", ...
				reduce_method: str. values can be "best", "weighted_sum", ...
				lambda_: float. weight for mean prototypes in "weighted_sum" reduce_method
				rho: float. weight for instance prototypes when update mean prototypes
		Returns:
			prediction: (B, H, W), prediction for each pixel

		"""
		multi_pred, to_logs = None, {}
		y = F.interpolate(y.unsqueeze(1).float(), size=outputs.shape[2:], mode="nearest").squeeze(1).long()
		proto_label = self.high_confident_proto_label(
			outputs, 
			threshold=cfg.confidence_threshold, 
		)
		proto, exists_flag = self.cal_proto(feature, proto_label)
		# init or update mean prototypes
		if not hasattr(self, "classifier_running_proto"):
			self.classifier_running_proto, self.classifier_running_proto_exists_flag = proto, exists_flag
		else:
			self.classifier_running_proto, self.classifier_running_proto_exists_flag = self._update_classifier_proto(feature, y, self.classifier_running_proto, self.classifier_running_proto_exists_flag, cfg.proto_rho)
		# make prediction
		instance_pred = self.cal_pred_of_proto(feature, proto, exists_flag)
		mean_pred = self.cal_pred_of_proto(feature, self.classifier_running_proto, self.classifier_running_proto_exists_flag)
		multi_pred = cfg.proto_lambda * mean_pred + (1-cfg.proto_lambda) * instance_pred
		return multi_pred, to_logs
	
	def cal_pred_of_proto(self, feature, proto, exists_flag, tau=1.0):
		""" Calculate the weight for classes of each pixel.
		For each pixel, we calculate the distance between the prototype of each class and the feature of the pixel.
		Then, we calculate the weight of each class by softmax with temperature tau.
		Note that we only calculate the weight of the classes that have prototypes, for other classes, the weight
		should be 0.
		Args:
			feature: [B, CH, H, W]
			proto: [C, CH]
			exists_flag: [C]
			tau: temperature
		Returns:
			weight: [B, C, H, W]
		"""
		# calculate distance
		feature = feature.unsqueeze(1) # [B, 1, CH, H, W]
		proto = proto.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # [1, C, CH, 1, 1]
		dist = torch.norm(proto - feature, dim=2) # [B, C, H, W]
		# calculate weight
		weight = torch.exp(-dist / tau) # [B, C, H, W]
		weight = weight * exists_flag.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # [B, C, H, W]
		weight = weight / torch.sum(weight, dim=1, keepdim=True) # [B, C, H, W]
		return weight

	@torch.no_grad()
	def _update_classifier_proto(self, feature, y, classifier_proto, classifier_proto_exists_flag, rho=None):
		"""Update the prototype of the classifier.
		Args:
			feature: [B, CH, H, W]
			y: [B, H, W]
		Returns:
			proto: [C, CH]
		"""
		# clone classifier proto and exists flag
		classifier_proto, classifier_proto_exists_flag = classifier_proto.clone(), classifier_proto_exists_flag.clone()
		# calculate mean of each class
		""" Update prototype for each class
		At first, we need to calculate the mean of each class and numbers of pixels of each class.
		Then, we can update the prototype of each class by total ratio.
		Args:
			feature: [B, C, H, W]
			y: [B, H, W]
		"""
		# calculate mean of each class
		proto, with_flag = self.cal_proto(feature, y)
		# update (set for the first time, update for the rest) 
		for i in range(self.hparams.dataset_info["num_classes"]):
			if not with_flag[i] or rho == 0: continue
			if classifier_proto_exists_flag[i] == 0:
				classifier_proto[i] = proto[i]
				classifier_proto_exists_flag[i] = 1
			else:
				classifier_proto[i] = (1-rho) * classifier_proto[i] + rho * proto[i]
		return classifier_proto, classifier_proto_exists_flag 

	@torch.no_grad()
	def proto_weight(self, proto, feature, tau=1.0):
		""" Calculate the weight for classes of each pixel.
		For each pixel, we calculate the distance between the prototype of each class and the feature of the pixel.
		Then, we calculate the weight of each class by softmax with temperature tau.
		Args:
			protos: [C, CH]
			feature: [B, CH, H, W]
			tau: temperature
		Returns:
			weight: [B, C, H, W]
		"""
		# calculate distance
		# [B, C, CH, H, W]
		proto = proto.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
		feature = feature.unsqueeze(1)
		dist = torch.norm(proto - feature, dim=2)
		# calculate weight
		weight = torch.softmax(-dist / tau, dim=1)
		return weight

	def cal_proto(self, feature, y):
		""" Calculate the prototype of each class with the given feature and label.
		Args:
			feature: [B, C, H, W]
			y: [B, H, W]
		Returns:
			proto: [C, C] # prototype of each class
			with_flag: [C] # whether the class has pixels
		"""
		proto = torch.zeros(self.hparams.dataset_info["num_classes"], feature.shape[1]).to(self.device)
		with_flag = torch.zeros(self.hparams.dataset_info["num_classes"]).to(self.device)
		for i in range(self.hparams.dataset_info["num_classes"]):
			masks = (y == i).flatten()
			if masks.sum() == 0:
				continue
			with_flag[i] = 1
			proto[i] = feature.permute((1, 0, 2, 3)).flatten(1).permute((1,0))[masks].mean(0)
		return proto, with_flag
	
	# utils operation
	def _configure_bn_running_stats(self):
		"""Configure model for use with eata."""
		for m in self.net.modules():
			if isinstance(m, nn.BatchNorm2d):
				m.lambda_.data = torch.tensor(self.hparams.cfg.bn_lambda)
	
	def _replace_bn(self):
		"""
		Replace all BN layers with new class for DIGA
		"""
		def get_layer(model, name):
			layer = model
			for attr in name.split("."):
				layer = getattr(layer, attr)
			return layer
		def set_layer(model, name, layer):
			try:
				attrs, name = name.rsplit(".", 1)
				model = get_layer(model, attrs)
			except ValueError:
				pass
			setattr(model, name, layer)
		# use replace all batch norm module m in self.net with custom batch norm module
		for n, module in self.net.named_modules():
			if isinstance(module, nn.BatchNorm2d):
				set_layer(self.net, n, SIFABatchNorm2dTrainable().from_bn(module).to(self.device))
	
	# others
	@property
	def wandb(self):
		for lg in self.loggers: 
			if "wandb" in lg.__module__:
				return lg
		raise ValueError("No wandb logger found")


if __name__ == "__main__":
	import hydra
	import omegaconf
	import pyrootutils

	root = pyrootutils.setup_root(__file__, pythonpath=True)
	cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "mnist.yaml")
	_ = hydra.utils.instantiate(cfg)
