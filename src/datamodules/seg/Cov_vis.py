import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
def get_covariance(tensor):
    bn, nk, w, h = tensor.shape
    tensor_reshape = tensor.reshape(bn, nk, 2, -1)
    x = tensor_reshape[:, :, 0, :]
    y = tensor_reshape[:, :, 1, :]
    mean_x = torch.mean(x, dim=2).unsqueeze(-1)
    mean_y = torch.mean(y, dim=2).unsqueeze(-1)

    xx = torch.sum((x - mean_x) * (x - mean_x), dim=2).unsqueeze(-1) / (h*w/2 - 1)
    xy = torch.sum((x - mean_x) * (y - mean_y), dim=2).unsqueeze(-1) / (h*w/2 - 1)
    yx = xy
    yy = torch.sum((y - mean_y) * (y - mean_y), dim=2).unsqueeze(-1) / (h*w/2 - 1)

    cov = torch.cat((xx, xy, yx, yy), dim=2)
    cov = cov.reshape(bn, nk, 2, 2)
    return cov

a = torch.randn([512, 512])

#a = (a-a.mean())/a.std()
b = torch.cov(a)
# b = (b-b.min())/(b.max()-b.min())
# a = a.detach().squeeze().numpy()

plt.scatter(b[:,1],b[:,2])
plt.show()

cv2.imshow('cov',b.detach().squeeze().numpy())
cv2.waitKey()





