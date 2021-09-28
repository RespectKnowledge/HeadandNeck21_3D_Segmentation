# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 19:47:37 2021

@author: Abdul Qayyum
"""
#%% Inception 3D unet
import os
import argparse
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable

class depthwise(nn.Module):
    '''
    depthwise convlution
    '''
    def __init__(self, cin, cout, kernel_size=3, stride=1, padding=3, dilation=1, depth=False):
        super(depthwise, self).__init__()
        if depth:
                self.Conv=nn.Sequential(OrderedDict([('conv1_1_depth', nn.Conv3d(cin, cin,
                        kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=cin)),
                        ('conv1_1_point', nn.Conv3d(cin, cout, 1))]))
        else:
            if stride>=1:
                self.Conv=nn.Conv3d(cin, cout, kernel_size=kernel_size, stride=stride,
                                                            padding=padding, dilation=dilation)
            else:
                stride = int(1//stride)
                self.Conv = nn.ConvTranspose3d(cin, cout, kernel_size=kernel_size, stride=stride,
                                                            padding=padding, dilation=dilation)
    def forward(self, x):
        return self.Conv(x)

class Inceptionblock_3D(nn.Module):
    def __init__(self, cin, co, relu=True, norm=True, normmethod = 'in',depth = True):
        super(Inceptionblock_3D, self).__init__()
        if normmethod == 'bn':
            Norm = nn.BatchNorm3d
        elif normmethod == 'in':
            Norm = nn.InstanceNorm3d
        else:
            print('please choose the correct normilze method!!!')
            os._exit()
        assert (co % 4 == 0)
        cos = [int(co / 4)] * 4
        self.activa = nn.Sequential()
        if norm: self.activa.add_module('norm', Norm(co))
        if relu: self.activa.add_module('relu', nn.ReLU())

        self.branch1 = nn.Conv3d(cin, cos[0], 1)

        self.branch2 = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv3d(cin, 2 * cos[1], 1)),
            ('norm1_1', Norm(2*cos[1])),
            ('relu1_1', nn.ReLU()),
            ('conv1_2_depth', depthwise(2 * cos[1], cos[1], 3, stride=1, padding=1, depth=depth)),
        ]))
        self.branch3 = nn.Sequential(OrderedDict([
            ('conv2_1', nn.Conv3d(cin, 2 * cos[2], 1, stride=1)),
            ('norm2_1', Norm(2*cos[2])),
            ('relu2_1', nn.ReLU()),
            ('conv2_2_depth', depthwise(2 * cos[2], cos[2], 5, stride=1, padding=2, depth=depth)),
        ]))

        self.branch4 = nn.Sequential(OrderedDict([
            ('pool3', nn.MaxPool3d(3, stride=1, padding=1)),
            ('conv3_1', nn.Conv3d(cin, cos[3], 1, stride=1))
        ]))

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        result = torch.cat((branch1, branch2, branch3, branch4), 1)
        return self.activa(result)

class SELayer3D(nn.Module):
    def __init__(self, channel):
        super(SELayer3D, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, round(channel/2), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(round(channel/2), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)
        
class SingleResConv3D(nn.Module):
    def __init__(self, cin, cout, norm='in', pad=1, depth=True, dilat=1):
        super(SingleResConv3D, self).__init__()
        if norm =='bn':
            Norm = nn.BatchNorm3d
        elif norm == 'in':
            Norm = nn.InstanceNorm3d
        else:
            raise ValueError('please choose the correct normilze method!!!')
        self.dilation = [[1, dilat, dilat], [dilat, 1, 1]]
        self.Input = nn.Conv3d(cin, cout, 1)
        self.norm = Norm(cout)
        self.active = nn.ReLU()
        if pad =='same':
            self.padding = dilat
        else:
            self.padding = pad
        self.model = nn.Sequential(OrderedDict([
            ('conv1_1', depthwise(cin, cout, 3, padding=self.padding, depth=depth, dilation=dilat)),
            ('norm1_1', Norm(cout)),
            ('relu1_1', nn.ReLU()),
        ]))

    def forward(self, x):

        return self.active(self.norm(self.model(x)+self.Input(x)))
        
class DoubleResConv3D(nn.Module):
    def __init__(self, cin, cout, norm='in', droprate=0, depth=True):
        super(DoubleResConv3D, self).__init__()
        if norm == 'bn':
            Norm = nn.BatchNorm3d
        elif norm == 'in':
            Norm = nn.InstanceNorm3d
        else:
            print('please choose the correct normilze method!!!')
            os._exit()
        self.Input = nn.Conv3d(cin, cout, 1)
        self.model = nn.Sequential(OrderedDict([
            ('conv1_1_depth', depthwise(cin, cout, 3, padding=1, depth=depth, dilation=1)),
            ('drop1_1', nn.Dropout(droprate)),
            ('norm1_1', Norm(cout)),
            ('relu1_1', nn.ReLU()),
            ('conv1_2_depth', depthwise(cout, cout, 3, padding=1, depth=depth, dilation=1)),
            ('drop1_2', nn.Dropout(droprate)),
            ('norm1_2', Norm(cout)),
            ('relu1_2', nn.ReLU()),
        ]))
        self.norm = Norm(cout)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.model(x)+self.Input(x)
        out = self.norm(out)
        return self.activation(out)

    
#%

import torch
from torch import nn
from torch.nn import functional as F


class SingleConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(SingleConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x, inplace=True)
        return x


n_cls: 2  # number of classes to predict (background and tumor)
in_channels: 2  # number of input modalities
n_filters: 24  # number of filters after the input (24 was used in the paper)

    
class ResIncepUNet(nn.Module):
    def __init__(self, in_channels, n_class, n_filters):
        super(ResIncepUNet, self).__init__()
        self.in_channels = in_channels
        self.n_class = 1 if n_class == 2 else n_class
        self.n_filters = n_filters
        
        # encoder block1

        self.b_1_1_enco = SingleConv3d(in_channels, n_filters, kernel_size=3, stride=1, padding=1)
        self.b_1_2_enco = SingleConv3d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.b_1_2_res1=DoubleResConv3D(n_filters, n_filters)
        self.b_1_2_se1=SELayer3D(n_filters)
        #self.b_1_2_incep1=Inception_v2_3D(n_filters, n_filters)
        
         # encoder block2

        self.p_1 = nn.MaxPool3d(kernel_size=2, stride=2)  # 64, 1/2
        self.b_2_1_enco = SingleConv3d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.b_2_2_enco = SingleConv3d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.b_2_2_res2=DoubleResConv3D(2 * n_filters, 2 * n_filters)
        self.b_2_2_se2=SELayer3D(2 * n_filters)
        
         # encoder block3

        self.p_2 = nn.MaxPool3d(kernel_size=2, stride=2)  # 128, 1/4
        self.b_3_1_enco = SingleConv3d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.b_3_2_enco = SingleConv3d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.b_3_2_res3=DoubleResConv3D(4 * n_filters, 4 * n_filters)
        self.b_3_2_se3=SELayer3D(4 * n_filters)
         
        # encoder block4

        self.p_3 = nn.MaxPool3d(kernel_size=2, stride=2)  # 256, 1/8
        self.b_4_1_enco = SingleConv3d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.b_4_2_enco = SingleConv3d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.b_4_2_res4=DoubleResConv3D(8 * n_filters, 8 * n_filters)
        self.b_4_2_se4=SELayer3D(8 * n_filters)
        
         # decoder block1

        self.upconv_3 = nn.ConvTranspose3d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        self.b_3_1_deco = SingleConv3d((4 + 4) * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.b_3_1_incep1deco=Inceptionblock_3D(4 * n_filters, 4 * n_filters)
        self.b_3_2_deco = SingleConv3d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        
        #decoder block2

        self.upconv_2 = nn.ConvTranspose3d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.b_2_1_deco = SingleConv3d((2 + 2) * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.b_2_1_incep1deco=Inceptionblock_3D(2 * n_filters, 2 * n_filters)
        self.b_2_2_deco = SingleConv3d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        
        #decoder block3

        self.upconv_1 = nn.ConvTranspose3d(2 * n_filters, n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.b_1_1_deco = SingleConv3d((1 + 1) * n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.b_1_1_incep1deco=Inceptionblock_3D(n_filters, n_filters)
        self.b_1_2_deco = SingleConv3d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        
        #last block

        self.conv1x1 = nn.Conv3d(n_filters, self.n_class, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        
        #encoder block1
        x=self.b_1_1_enco(x)
        x=self.b_1_2_enco(x)
        x=self.b_1_2_res1(x)
        ds0=self.b_1_2_se1(x)
        #ds0=self.b_1_2_incep1(x)
        #print(ds0.shape)
        #encoder block2
        x1=self.p_1(ds0)
        x1=self.b_2_1_enco(x1)
        x1=self.b_2_2_enco(x1)
        x1=self.b_2_2_res2(x1)
        ds1=self.b_2_2_se2(x1)
        #print(ds1.shape)
        #encoder block3
        x2=self.p_2(ds1)
        x2=self.b_3_1_enco(x2)
        x2=self.b_3_2_enco(x2)
        x2=self.b_3_2_res3(x2)
        ds2=self.b_3_2_se3(x2)
        #print(ds2.shape)
        #encoder block4
        x3=self.p_3(ds2)
        x3=self.b_4_1_enco(x3)
        x3=self.b_4_2_enco(x3)
        x3=self.b_4_2_res4(x3)
        x=self.b_4_2_se4(x3)
        
        #print(x.shape)
        #decoder 1
        d1=torch.cat([self.upconv_3(x), ds2], 1)
        d1=self.b_3_1_incep1deco(d1)
        d1=self.b_3_1_deco(d1)
        x=self.b_3_2_deco(d1)
        #decoder2
        d2=torch.cat([self.upconv_2(x), ds1], 1)
        d2=self.b_2_1_deco(d2)
        d2=self.b_2_1_incep1deco(d2)
        x=self.b_2_2_deco(d2)
        #decoder 3
        d3=torch.cat([self.upconv_1(x), ds0], 1)
        d3=self.b_1_1_deco(d3)
        #d3=self.b_1_1_incep1deco(d3)
        x=self.b_1_2_deco(d3)
        #1x1 conv last
        x = self.conv1x1(x)

        if self.n_class == 1:
            return torch.sigmoid(x)
        else:
            return F.softmax(x, dim=1)
# test the model
inp=torch.rand(1,1,144,144,144)   
model= ResIncepUNet(2,1,24)
out=model(inp)
print(out.shape)


class ResIncepUNet5(nn.Module):
    def __init__(self, in_channels, n_class, n_filters):
        super(ResIncepUNet5, self).__init__()
        self.in_channels = in_channels
        self.n_class = 1 if n_class == 2 else n_class
        self.n_filters = n_filters
        
        # encoder block1

        self.b_1_1_enco = SingleConv3d(in_channels, n_filters, kernel_size=3, stride=1, padding=1)
        self.b_1_2_enco = SingleConv3d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.b_1_2_res1=DoubleResConv3D(n_filters, n_filters)
        self.b_1_2_incep1deco=Inceptionblock_3D(n_filters, n_filters)
        self.b_1_2_se1=SELayer3D(n_filters)
        #self.b_1_2_incep1=Inception_v2_3D(n_filters, n_filters)
        
         # encoder block2

        self.p_1 = nn.MaxPool3d(kernel_size=2, stride=2)  # 64, 1/2
        self.b_2_1_enco = SingleConv3d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.b_2_2_enco = SingleConv3d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.b_2_2_res2=DoubleResConv3D(2 * n_filters, 2 * n_filters)
        self.b_2_2_incep1deco=Inceptionblock_3D(2*n_filters, 2*n_filters)
        self.b_2_2_se2=SELayer3D(2 * n_filters)
        
         # encoder block3

        self.p_2 = nn.MaxPool3d(kernel_size=2, stride=2)  # 128, 1/4
        self.b_3_1_enco = SingleConv3d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.b_3_2_enco = SingleConv3d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.b_3_2_res3=DoubleResConv3D(4 * n_filters, 4 * n_filters)
        self.b_3_2_incep1deco=Inceptionblock_3D(4*n_filters, 4*n_filters)
        self.b_3_2_se3=SELayer3D(4 * n_filters)
         
        # encoder block4

        self.p_3 = nn.MaxPool3d(kernel_size=2, stride=2)  # 256, 1/8
        self.b_4_1_enco = SingleConv3d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.b_4_2_enco = SingleConv3d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.b_4_2_res4=DoubleResConv3D(8 * n_filters, 8 * n_filters)
        self.b_4_2_incep1deco=Inceptionblock_3D(8*n_filters, 8*n_filters)
        self.b_4_2_se4=SELayer3D(8 * n_filters)
        # encoder block5
        self.p_4 = nn.MaxPool3d(kernel_size=2, stride=2)  # 256, 1/8
        self.b_5_1_enco = SingleConv3d(8 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)
        self.b_5_2_enco = SingleConv3d(16 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)
        self.b_5_2_res4=DoubleResConv3D(16 * n_filters, 16 * n_filters)
        self.b_5_2_incep1deco=Inceptionblock_3D(16*n_filters, 16*n_filters)
        self.b_5_2_se4=SELayer3D(16 * n_filters)
        
        # decoder block1
        
        self.upconv_4 = nn.ConvTranspose3d(16 * n_filters, 8 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        self.b_4_1_deco = SingleConv3d((8 + 8) * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.b_4_1_incep1deco=Inceptionblock_3D(8 * n_filters, 8 * n_filters)
        self.b_4_1_res=DoubleResConv3D(8 * n_filters, 8 * n_filters)
        self.b_4_2_deco = SingleConv3d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        #self.vision_4 = UpConv(8 * n_filters, n_filters, 2, scale=8)
        
         # decoder block1

        self.upconv_3 = nn.ConvTranspose3d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        self.b_3_1_deco = SingleConv3d((4 + 4) * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.b_3_1_incep1deco=Inceptionblock_3D(4 * n_filters, 4 * n_filters)
        self.b_3_1_res=DoubleResConv3D(4 * n_filters, 4 * n_filters)
        self.b_3_2_deco = SingleConv3d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        #self.vision_3 = UpConv(4 * n_filters, n_filters, 2, scale=4)
        
        #decoder block2

        self.upconv_2 = nn.ConvTranspose3d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.b_2_1_deco = SingleConv3d((2 + 2) * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.b_2_1_incep1deco=Inceptionblock_3D(2 * n_filters, 2 * n_filters)
        self.b_2_1_res=DoubleResConv3D(2 * n_filters, 2 * n_filters)
        self.b_2_2_deco = SingleConv3d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        #self.vision_2 = UpConv(2 * n_filters, n_filters, 2, scale=2)
        
        #decoder block3

        self.upconv_1 = nn.ConvTranspose3d(2 * n_filters, n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.b_1_1_deco = SingleConv3d((1 + 1) * n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.b_1_1_incep1deco=Inceptionblock_3D(n_filters, n_filters)
        self.b_1_1_res=DoubleResConv3D(n_filters, n_filters)
        self.b_1_2_deco = SingleConv3d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        #self.vision_2 = UpConv(2 * n_filters, n_filters, 2, scale=2)
        
        #last block

        self.conv1x1 = nn.Conv3d(n_filters, self.n_class, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        
        #encoder block1
        x=self.b_1_1_enco(x)
        x=self.b_1_2_enco(x)
        #x=self.b_1_2_res1(x)
        x=self.b_1_2_incep1deco(x)
        ds0=self.b_1_2_se1(x)
        #ds0=self.b_1_2_incep1(x)
        #print(ds0.shape)
        #encoder block2
        x1=self.p_1(ds0)
        x1=self.b_2_1_enco(x1)
        x1=self.b_2_2_enco(x1)
        #x1=self.b_2_2_res2(x1)
        x1=self.b_2_2_incep1deco(x1)
        ds1=self.b_2_2_se2(x1)
        #print(ds1.shape)
        #encoder block3
        x2=self.p_2(ds1)
        x2=self.b_3_1_enco(x2)
        x2=self.b_3_2_enco(x2)
        #x2=self.b_3_2_res3(x2)
        x2=self.b_3_2_incep1deco(x2)
        ds2=self.b_3_2_se3(x2)
        #print(ds2.shape)
        #encoder block4
        x3=self.p_3(ds2)
        x3=self.b_4_1_enco(x3)
        x3=self.b_4_2_enco(x3)
        #x3=self.b_4_2_res4(x3)
        x3=self.b_4_2_incep1deco(x3)
        ds3=self.b_4_2_se4(x3)
        
        x4=self.p_4(ds3)
        x4=self.b_5_1_enco(x4)
        #x4=self.b_5_2_res4(x4)
        x4=self.b_5_2_enco(x4)
        #x4=self.b_5_2_res4(x4)
        x4=self.b_5_2_incep1deco(x4)
        x=self.b_5_2_se4(x4)
        
        #decoder 1
        d1=torch.cat([self.upconv_4(x), ds3], 1)
        #d1=self.b_3_1_incep1deco(d1)
        d1=self.b_4_1_deco(d1)
        d1=self.b_4_1_res(d1)
        x=self.b_4_2_deco(d1)
        # print('xlast',x.shape)
        # sv4=self.vision_4(x)
        # print('sv4',sv4.shape)
        
        #print(x.shape)
        #decoder 1
        d2=torch.cat([self.upconv_3(x), ds2], 1)
        #d1=self.b_3_1_incep1deco(d1)
        d2=self.b_3_1_deco(d2)
        d2=self.b_3_1_res(d2)
        x=self.b_3_2_deco(d2)
        # print('xlast',x.shape)
        # sv4=self.vision_4(x)
        # print('sv4',sv4.shape)
        
        #decoder2
        d3=torch.cat([self.upconv_2(x), ds1], 1)
        d3=self.b_2_1_deco(d3)
        #d2=self.b_2_1_incep1deco(d2)
        d3=self.b_2_1_res(d3)
        x=self.b_2_2_deco(d3)
        # print('2ndlast',x.shape)
        # sv3=self.vision_3(x)
        # print('sv3',sv3.shape)
        #decoder 3
        d4=torch.cat([self.upconv_1(x), ds0], 1)
        d4=self.b_1_1_deco(d4)
        d4=self.b_1_1_res(d4)
        #d3=self.b_1_1_incep1deco(d3)
        x=self.b_1_2_deco(d4)
        # print('3rdlast',x)
        # sv2=self.vision_2(x)
        # print('sv2',sv2.shape)
        #1x1 conv last
        x = self.conv1x1(x)

        if self.n_class == 1:
            return torch.sigmoid(x)
        else:
            return F.softmax(x, dim=1)
# test the model
inp=torch.rand(1,1,144,144,144)              
model= ResIncepUNet5(2,1,24)
out=model(inp)
print(out.shape)