# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 14:53:57 2021

@author: Abdul Qayyum
"""

#%% Prediction function devloped by the challenege organizer
#%ResNet3D deeper 
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
    def __init__(self, cin, co, relu=True, norm=True, normmethod = 'in',depth = False):
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
    def __init__(self, cin, cout, norm='in', pad=1, depth=False, dilat=1):
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
    def __init__(self, cin, cout, norm='in', droprate=0, depth=False):
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
class ResIncepUNet4(nn.Module):
    def __init__(self, in_channels, n_class, n_filters):
        super(ResIncepUNet4, self).__init__()
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
        self.b_5_2_res5=DoubleResConv3D(16 * n_filters, 16 * n_filters)
        self.b_5_2_incep1deco=Inceptionblock_3D(16*n_filters, 16*n_filters)
        self.b_5_2_se5=SELayer3D(16 * n_filters)
        
        # decoder block1
        
        self.upconv_4 = nn.ConvTranspose3d(16 * n_filters, 8 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        self.b_4_1_deco = SingleConv3d((8 + 8) * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.b_4_1_incep1deco=Inceptionblock_3D(8 * n_filters, 8 * n_filters)
        self.b_4_1_res=DoubleResConv3D(8 * n_filters, 8 * n_filters)
        self.b_4_2_deco = SingleConv3d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.vision_4 = UpConv(8 * n_filters, n_filters, 2, scale=8)
        
         # decoder block1

        self.upconv_3 = nn.ConvTranspose3d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        self.b_3_1_deco = SingleConv3d((4 + 4) * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.b_3_1_incep1deco=Inceptionblock_3D(4 * n_filters, 4 * n_filters)
        self.b_3_1_res=DoubleResConv3D(4 * n_filters, 4 * n_filters)
        self.b_3_2_deco = SingleConv3d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.vision_3 = UpConv(4 * n_filters, n_filters, 2, scale=4)
        
        #decoder block2

        self.upconv_2 = nn.ConvTranspose3d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.b_2_1_deco = SingleConv3d((2 + 2) * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.b_2_1_incep1deco=Inceptionblock_3D(2 * n_filters, 2 * n_filters)
        self.b_2_1_res=DoubleResConv3D(2 * n_filters, 2 * n_filters)
        self.b_2_2_deco = SingleConv3d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.vision_2 = UpConv(2 * n_filters, n_filters, 2, scale=2)
        
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
        x=self.b_1_2_res1(x)
        #x=self.b_1_2_incep1deco(x)
        ds0=self.b_1_2_se1(x)
        #ds0=self.b_1_2_incep1(x)
        #print(ds0.shape)
        #encoder block2
        x1=self.p_1(ds0)
        x1=self.b_2_1_enco(x1)
        x1=self.b_2_2_enco(x1)
        x1=self.b_2_2_res2(x1)
        #x1=self.b_2_2_incep1deco(x1)
        ds1=self.b_2_2_se2(x1)
        #print(ds1.shape)
        #encoder block3
        x2=self.p_2(ds1)
        x2=self.b_3_1_enco(x2)
        x2=self.b_3_2_enco(x2)
        x2=self.b_3_2_res3(x2)
        #x2=self.b_3_2_incep1deco(x2)
        ds2=self.b_3_2_se3(x2)
        #print(ds2.shape)
        #encoder block4
        x3=self.p_3(ds2)
        x3=self.b_4_1_enco(x3)
        x3=self.b_4_2_enco(x3)
        x3=self.b_4_2_res4(x3)
        #x3=self.b_4_2_incep1deco(x3)
        ds3=self.b_4_2_se4(x3)
        
        x4=self.p_4(ds3)
        x4=self.b_5_1_enco(x4)
        x4=self.b_5_2_enco(x4)
        x4=self.b_5_2_res5(x4)
        #x3=self.b_4_2_incep1deco(x3)
        x=self.b_5_2_se5(x4)
        
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
model= ResIncepUNet4(2,1,24)

#%load model and test dataset for prediction
import torch
import numpy as np
import nibabel as nib
import torch 
from pathlib import Path
import os
import sys
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import random
from skimage.transform import rotate
############ give the path of trained model
path1="C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\Hector2021\\hecktor2021_train\\hecktor2021_train\\trained_models\\model_3DResSEDeeper.pth"
model.load_state_dict(torch.load(path1))
model.eval()


class Compose:
    def __init__(self, transforms=None):
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)

        return sample

    
    
class ToTensor2:  
    def __call__(self, sample):
        img= sample,
        img = np.transpose(img, axes=[3, 0, 1, 2])
        img = torch.from_numpy(img).float()
        return img


class NormalizeIntensity:

    def __call__(self, sample):
        img = sample
        img[:, :, :, 0] = self.normalize_ct(img[:, :, :, 0])
        img[:, :, :, 1] = self.normalize_pt(img[:, :, :, 1])

        img
        return img

    @staticmethod
    def normalize_ct(img):
        norm_img = np.clip(img, -1024, 1024) / 1024
        return norm_img

    @staticmethod
    def normalize_pt(img):
        mean = np.mean(img)
        std = np.std(img)
        return (img - mean) / (std + 1e-3)
    
    
#%% Prediction function devloped by the challenege organizer
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
############ provide the path of test dataset and output folder 
# output folder will be use to store the result
test_folder = Path("C:\\Users\\Administrateur\\Desktop\\micca2021\MICCAI2021\\Hector2021\\testing2021hector\\hecktor2021_test\\hecktor2021_test\\hecktor_nii").resolve()
results_folder = Path("C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\Hector2021\\testing2021hector\\segmentation_results_ResNetSE\\").resolve()
#results_folder.mkdir(exist_ok=True)

bbox_df = pd.read_csv("C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\Hector2021\\testing2021hector\\hecktor2021_test\\hecktor2021_test\\hecktor2021_bbox_testing.csv").set_index("PatientID")


def dummy_model(x):
    return np.random.uniform(size=x.shape[:4] + (1, ))

def normstackctandpet(image_ct,image_pt):
    #ct_img_aff=nib.load(str(pathct)).affine
    arrayct = np.transpose(sitk.GetArrayFromImage(image_ct), (2, 1, 0))
    pt_img = np.transpose(sitk.GetArrayFromImage(image_pt), (2, 1, 0))

    #ct_img=nib.load(str(pathct)).get_fdata()
    norm_img = np.clip(arrayct, -1024, 1024) / 1024

    # pt_img=nib.load(str(pathpt)).get_fdata()
    # array_ct = np.transpose(sitk.GetArrayFromImage(ct_img), (2, 1, 0))
    # array_pt = np.transpose(sitk.GetArrayFromImage(image_pt), (2, 1, 0))

    
    mean = np.mean(pt_img)
    std = np.std(pt_img)
    pt_img_n=(pt_img - mean) / (std + 1e-3)
    imgc=[norm_img,pt_img_n] # normalize 
    img = np.stack(imgc, axis=-1) #stacking two channels

    img = np.transpose(img, axes=[3, 0, 1, 2])
    img_t = torch.from_numpy(img).float()
    img_ts = img_t.unsqueeze(0)
    return img_ts

def prediction(img_ts,model):
    with torch.no_grad():
        model.eval()
        model.cuda()
        output=model(img_ts)
        mask = output.detach().cpu().numpy()
        output1 = (mask > 0.5).astype(int)  # get binary label
        print(np.unique(output1))
        out=np.squeeze(np.squeeze(output1, axis=0),axis=0)
    return out

patient_list = [f.name[:7] for f in test_folder.rglob("*_ct.nii.gz")]

# Instantiating the resampler
resampling_spacing = np.array([1.0, 1.0, 1.0])
pre_resampler = sitk.ResampleImageFilter()
pre_resampler.SetInterpolator(sitk.sitkBSpline)
pre_resampler.SetOutputSpacing(resampling_spacing)

post_resampler = sitk.ResampleImageFilter()
post_resampler.SetInterpolator(sitk.sitkNearestNeighbor)

for p_id in tqdm(patient_list):
    # loading the images and storing the ct spacing
    image_ct = sitk.ReadImage(str(test_folder / (p_id + "_ct.nii.gz")))
    image_pt = sitk.ReadImage(str(test_folder / (p_id + "_pt.nii.gz")))
    spacing_ct = image_ct.GetSpacing()

    # getting the bounding box
    bb = np.squeeze(
        np.array([
            bbox_df.loc[p_id, ["x1", "y1", "z1", "x2", "y2", "z2"]],
        ]))

    # resampling the images
    resampled_size = np.round(
        (bb[3:] - bb[:3]) / resampling_spacing).astype(int)
    pre_resampler.SetOutputOrigin(bb[:3])
    pre_resampler.SetSize([int(k)
                           for k in resampled_size])  # sitk requires this
    image_ct = pre_resampler.Execute(image_ct)
    image_pt = pre_resampler.Execute(image_pt)

    # sitk to numpy, sitk stores images with [dim_z, dim_y, dim_x]
    # array_ct = np.transpose(sitk.GetArrayFromImage(image_ct), (2, 1, 0))
    # array_pt = np.transpose(sitk.GetArrayFromImage(image_pt), (2, 1, 0))

    # ... apply your preprocessing here

    # x = np.stack([array_ct, array_pt], axis=0)
    # x = x[np.newaxis, ...]  # adding batch dimension
    # segmentation = dummy_model(x)[0, :, :, :, 0]

    # # do not forget to threshold your output
    # segmentation = (segmentation < 0.5).astype(np.uint8)
    
    img_ts=normstackctandpet(image_ct,image_pt)
    img_ts=img_ts.cuda()
    #print(img_ts.shape)
    segmentation=prediction(img_ts,model)

    # numpy to sitk
    image_segmentation = sitk.GetImageFromArray(
        np.transpose(segmentation, (2, 1, 0)))

    image_segmentation.SetOrigin(bb[:3])
    image_segmentation.SetSpacing(resampling_spacing)

    # If you do not resample to the orginal CT resolution,
    # the following nearest neighbor resampling will be applied to your submission.
    # We encourage you to try other resampling methods that are more suited to
    # binary mask.
    final_size = np.round((bb[3:] - bb[:3]) / spacing_ct).astype(int)
    post_resampler.SetOutputSpacing(spacing_ct)
    post_resampler.SetOutputOrigin(bb[:3])
    post_resampler.SetSize([int(k) for k in final_size])  # sitk requires this

    image_segmentation = post_resampler.Execute(image_segmentation)

    # Saving the prediction
    sitk.WriteImage(image_segmentation,str(results_folder / (p_id + ".nii.gz")),
                    )
#%% prediction dice and hd from hector dataset
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

#from src.evaluation.scores import dice, hausdorff_distance
import numpy as np
from scipy.spatial import cKDTree


def dice(mask_gt, mask_seg):
    return 2 * np.sum(np.logical_and(
        mask_gt, mask_seg)) / (np.sum(mask_gt) + np.sum(mask_seg))


def hausdorff_distance(image0, image1):
    """Code copied from 
    https://github.com/scikit-image/scikit-image/blob/main/skimage/metrics/set_metrics.py#L7-L54
    for compatibility reason with python 3.6
    """
    a_points = np.transpose(np.nonzero(image0))
    b_points = np.transpose(np.nonzero(image1))

    # Handle empty sets properly:
    # - if both sets are empty, return zero
    # - if only one set is empty, return infinity
    if len(a_points) == 0:
        return 0 if len(b_points) == 0 else np.inf
    elif len(b_points) == 0:
        return np.inf

    return max(max(cKDTree(a_points).query(b_points, k=1)[0]),
               max(cKDTree(b_points).query(a_points, k=1)[0]))


prediction_folder = 'C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\Hector2021\\hecktor2021_train\\hecktor2021_train\\predictions\\resunet3dse'
groundtruth_folder = 'C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\Hector2021\\hecktor2021_train\\hecktor2021_train\\hecktor_nii_resampled'
bb_filepath = 'C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\Hector2021\\hecktor2021_train\\hecktor2021_train\\hecktor2021_bbox_training.csv'
results='C:\\Users\\Administrateur\\Desktop\\micca2021\MICCAI2021\\Hector2021\\hecktor2021_train\\hecktor2021_train\\results'
# List of the files in the validation
prediction_files = [f for f in Path(prediction_folder).rglob('*.nii.gz')]

# The list is sorted, so it will match the list of ground truth files
prediction_files.sort(key=lambda x: x.name.split('_')[0])

# List of the patient_id in the validation
patient_name_predictions = [f.name.split('.')[0][:7] for f in prediction_files]


# List of the ground truth files
groundtruth_files = [
    f for f in Path(groundtruth_folder).rglob('*gtvt.nii.gz') if f.name.split('_')[0] in patient_name_predictions
]

# The bounding boxes will be used to compute the Dice score within.
bb_df = pd.read_csv(bb_filepath).set_index('PatientID')

# DataFrame to store the results
results_df = pd.DataFrame(columns=['PatientID', 'Dice Score'])

resampler = sitk.ResampleImageFilter()
resampler.SetInterpolator(sitk.sitkNearestNeighbor)

for f in prediction_files:
    patient_name = f.name.split('.')[0][:7]
    gt_file = [k for k in groundtruth_files if k.name[:7] == patient_name][0]

    print('Evaluating patient {}'.format(patient_name))

    sitk_pred = sitk.ReadImage(str(f.resolve()))
    sitk_gt = sitk.ReadImage(str(gt_file.resolve()))
    resampling_spacing = np.array(sitk_gt.GetSpacing())

    bb = np.array([
        bb_df.loc[patient_name, 'x1', ], bb_df.loc[patient_name, 'y1', ],
        bb_df.loc[patient_name, 'z1', ], bb_df.loc[patient_name, 'x2', ],
        bb_df.loc[patient_name, 'y2', ], bb_df.loc[patient_name, 'z2', ]
    ])

    image_size = np.round((bb[3:] - bb[:3]) / resampling_spacing).astype(int)
    resampler.SetOutputOrigin(bb[:3])
    resampler.SetSize([int(k) for k in image_size])
    resampler.SetReferenceImage(sitk_gt)

    sitk_gt = resampler.Execute(sitk_gt)
    sitk_pred = resampler.Execute(sitk_pred)

    # Store the results
    np_gt = sitk.GetArrayFromImage(sitk_gt)
    np_pred = sitk.GetArrayFromImage(sitk_pred)
    results_df = results_df.append(
        {
            'PatientID': patient_name,
            'Dice Score': dice(np_gt, np_pred),
            'Hausdorff Distance': hausdorff_distance(np_gt, np_pred),
        },
        ignore_index=True)
    
dcmean=np.mean(list(results_df['Dice Score']))
print('dcmean',dcmean)
hdmean=np.mean(list(results_df['Hausdorff Distance']))
print('hdmean',hdmean)
import os