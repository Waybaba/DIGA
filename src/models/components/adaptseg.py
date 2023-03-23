import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torchvision import models


affine_par = True


def outS(i):
    i = int(i)
    i = (i + 1) / 2
    i = int(np.ceil((i + 1) / 2.0))
    i = (i + 1) / 2
    return i


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
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


class Classifier_Module(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x, feat=False):
        if feat:
            feat = x
            out = self.conv2d_list[0](x) 
            for i in range(len(self.conv2d_list) - 1):
                out += self.conv2d_list[i + 1](x)
                return feat, out
        out = self.conv2d_list[0](x) 
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
            return out


class ResNetMulti(nn.Module):
    def __init__(self, block, layers, num_classes):
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
        self.layer5 = self._make_pred_layer(Classifier_Module, 1024, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.layer6 = self._make_pred_layer(Classifier_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                #        for i in m.parameters():
                #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)

    def forward(self, x, feat=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if feat:
            x2 = self.layer4(x)
            feat, x2 = self.layer6(x2, feat=feat)
            return feat, x2

        x1 = self.layer5(x)

        x2 = self.layer4(x)
        x2 = self.layer6(x2)
        return x1, x2

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

    # def optim_parameters(self, args):
    #     return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate},
    #             {'params': self.get_10x_lr_params(), 'lr': 10 * args.learning_rate}]

    def optim_parameters(self, lr):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': lr},
                {'params': self.get_10x_lr_params(), 'lr': 10 * lr}]

def DeeplabMulti(num_classes=21):
    model = ResNetMulti(Bottleneck, [3, 4, 23, 3], num_classes)
    return model


# class Classifier_Module(nn.Module):

#     def __init__(self, dims_in, dilation_series, padding_series, num_classes):
#         super(Classifier_Module, self).__init__()
#         self.conv2d_list = nn.ModuleList()
#         for dilation, padding in zip(dilation_series, padding_series):
#             self.conv2d_list.append(nn.Conv2d(dims_in, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias = True))

#         for m in self.conv2d_list:
#             m.weight.data.normal_(0, 0.01)

#     def forward(self, x):
#         out = self.conv2d_list[0](x)
#         for i in range(len(self.conv2d_list)-1):
#             out += self.conv2d_list[i+1](x)
#             return out

import numpy as np
import torch
from torch import nn
from torchvision import models
from torch.nn import functional as F

# class Classifier_Module(nn.Module):

# 	def __init__(self, dims_in, dilation_series, padding_series, num_classes):
# 		super(Classifier_Module, self).__init__()
# 		self.conv2d_list = nn.ModuleList()
# 		for dilation, padding in zip(dilation_series, padding_series):
# 			self.conv2d_list.append(nn.Conv2d(dims_in, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias = True))

# 		for m in self.conv2d_list:
# 			m.weight.data.normal_(0, 0.01)

# 	def forward(self, x):
# 		out = self.conv2d_list[0](x)
# 		for i in range(len(self.conv2d_list)-1):
# 			out += self.conv2d_list[i+1](x)
# 			return out

import os

class DeeplabVGG(nn.Module):

	def __init__(self, num_classes, vgg16_caffe_path=None, pretrained=False):
		super(DeeplabVGG, self).__init__()
		vgg = models.vgg16()
		# load pretrained model
		# model path $UDATADIR/models/DA_Seg_models/VGG/vgg16-00b39a1b-updated.pth
		DEFUALT_PATH = os.path.join(os.environ['UDATADIR'], 'models/DA_Seg_models/VGG/vgg16-00b39a1b-updated.pth')
		if vgg16_caffe_path is None:
			vgg16_caffe_path = DEFUALT_PATH
		if pretrained:
			vgg.load_state_dict(torch.load(vgg16_caffe_path))

		features, classifier = list(vgg.features.children()), list(vgg.classifier.children())

		#remove pool4/pool5
		features = nn.Sequential(*(features[i] for i in list(range(23))+list(range(24,30))))

		for i in [23,25,27]:
			features[i].dilation = (2,2)
			features[i].padding = (2,2)

		fc6 = nn.Conv2d(512, 1024, kernel_size=3, padding=4, dilation=4)
		fc7 = nn.Conv2d(1024, 1024, kernel_size=3, padding=4, dilation=4)

		self.features = nn.Sequential(*([features[i] for i in range(len(features))] + [ fc6, nn.ReLU(inplace=True), fc7, nn.ReLU(inplace=True)]))

		self.classifier = Classifier_Module(1024, [6,12,18,24],[6,12,18,24],num_classes)

	def forward(self, x):
		"""
		e.g. for resnet 101
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
		"""
		input_size = x.size()[2:]
		x = self.features(x)
		x = self.classifier(x)
		x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
		return x

	def get_1x_lr_params_NOscale(self):
		"""
		This generator returns all the parameters of the net except for
		the last classification layer. Note that for each batchnorm layer,
		requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
		any batchnorm parameter
		e.g. for resnet101
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
		"""
		b = []
		b.append(self.features.parameters())

		for j in range(len(b)):
			for i in b[j]:
				yield i

	def get_10x_lr_params(self):
		"""
		This generator returns all the parameters for the last layer of the net,
		which does the classification of pixel into classes
		e.g. for resnet 101
		b = []
		b.append(self.layer5.parameters())
		b.append(self.layer6.parameters())

		for j in range(len(b)):
			for i in b[j]:
				yield i
		"""
		b = []
		b.append(self.classifier.parameters())

		for j in range(len(b)):
			for i in b[j]:
				yield i

	def optim_parameters(self, args):
		return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.lr},
				{'params': self.get_10x_lr_params(), 'lr': 10 * args.lr}]
