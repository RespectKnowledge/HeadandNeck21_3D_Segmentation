# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 19:41:38 2021

@author: Abdul Qayyum
"""
#%% Basic 3D Unet model for hector Segmentation
import torch
from torch import nn
from torch.nn import functional as F


class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x, inplace=True)
        return x
  
class FastSmoothSENorm(nn.Module):
    class SEWeights(nn.Module):
        def __init__(self, in_channels, reduction=2):
            super().__init__()
            self.conv1 = nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0, bias=True)
            self.conv2 = nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0, bias=True)

        def forward(self, x):
            b, c, d, h, w = x.size()
            out = torch.mean(x.view(b, c, -1), dim=-1).view(b, c, 1, 1, 1)  # output_shape: in_channels x (1, 1, 1)
            out = F.relu(self.conv1(out))
            out = self.conv2(out)
            return out

    def __init__(self, in_channels, reduction=2):
        super(FastSmoothSENorm, self).__init__()
        self.norm = nn.InstanceNorm3d(in_channels, affine=False)
        self.gamma = self.SEWeights(in_channels, reduction)
        self.beta = self.SEWeights(in_channels, reduction)

    def forward(self, x):
        gamma = torch.sigmoid(self.gamma(x))
        beta = torch.tanh(self.beta(x))
        x = self.norm(x)
        return gamma * x + beta


class FastSmoothSeNormConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=2, **kwargs):
        super(FastSmoothSeNormConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=True, **kwargs)
        self.norm = FastSmoothSENorm(out_channels, reduction)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        x = self.norm(x)
        return x

 
class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=2, scale=2):
        super().__init__()
        self.scale = scale
        self.conv = FastSmoothSeNormConv3d(in_channels, out_channels, reduction, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv(x)
        x = F.interpolate(x, scale_factor=self.scale, mode='trilinear', align_corners=False)
        return x
    

n_cls: 2  # number of classes to predict (background and tumor)
in_channels: 2  # number of input modalities
n_filters: 24  # number of filters after the input (24 was used in the paper)
reduction: 2  # parameter controls the size of the bottleneck in SENorm layers

class BaselineUNet(nn.Module):
    def __init__(self, in_channels, n_cls, n_filters):
        super(BaselineUNet, self).__init__()
        self.in_channels = in_channels
        self.n_cls = 1 if n_cls == 2 else n_cls
        self.n_filters = n_filters

        self.block_1_1_left = BasicConv3d(in_channels, n_filters, kernel_size=3, stride=1, padding=1)
        self.block_1_2_left = BasicConv3d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_1 = nn.MaxPool3d(kernel_size=2, stride=2)  # 64, 1/2
        self.block_2_1_left = BasicConv3d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_2_2_left = BasicConv3d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_2 = nn.MaxPool3d(kernel_size=2, stride=2)  # 128, 1/4
        self.block_3_1_left = BasicConv3d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_3_2_left = BasicConv3d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2)  # 256, 1/8
        self.block_4_1_left = BasicConv3d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_4_2_left = BasicConv3d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_3 = nn.ConvTranspose3d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_3_1_right = BasicConv3d((4 + 4) * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_3_2_right = BasicConv3d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_2 = nn.ConvTranspose3d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_2_1_right = BasicConv3d((2 + 2) * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_2_2_right = BasicConv3d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_1 = nn.ConvTranspose3d(2 * n_filters, n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_1_1_right = BasicConv3d((1 + 1) * n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.block_1_2_right = BasicConv3d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.conv1x1 = nn.Conv3d(n_filters, self.n_cls, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        ds0 = self.block_1_2_left(self.block_1_1_left(x))
        ds1 = self.block_2_2_left(self.block_2_1_left(self.pool_1(ds0)))
        ds2 = self.block_3_2_left(self.block_3_1_left(self.pool_2(ds1)))
        x = self.block_4_2_left(self.block_4_1_left(self.pool_3(ds2)))

        x = self.block_3_2_right(self.block_3_1_right(torch.cat([self.upconv_3(x), ds2], 1)))
        x = self.block_2_2_right(self.block_2_1_right(torch.cat([self.upconv_2(x), ds1], 1)))
        x = self.block_1_2_right(self.block_1_1_right(torch.cat([self.upconv_1(x), ds0], 1)))

        x = self.conv1x1(x)

        if self.n_cls == 1:
            return torch.sigmoid(x)
        else:
            return F.softmax(x, dim=1)
inp=torch.rand(1,2,144,144,144)
model=BaselineUNet(2,2,24)
out=model(inp)
print(out.shape)
