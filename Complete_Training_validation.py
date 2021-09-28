# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 20:27:43 2021

@author: Abdul Qayyum
"""
#%% complete code for training and optimization
import torch 
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import os
import sys
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import nibabel as nib
import random
from skimage.transform import rotate


class Compose:
    def __init__(self, transforms=None):
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)

        return sample


class ToTensor:
    def __init__(self, mode='train'):
        if mode not in ['train', 'test']:
            raise ValueError(f"Argument 'mode' must be 'train' or 'test'. Received {mode}")
        self.mode = mode

    def __call__(self, sample):
        if self.mode == 'train':
            img, mask = sample['input'], sample['target']
            img = np.transpose(img, axes=[3, 0, 1, 2])
            mask = np.transpose(mask, axes=[3, 0, 1, 2])
            img = torch.from_numpy(img).float()
            mask = torch.from_numpy(mask).float()
            sample['input'], sample['target'] = img, mask

        else:  # if self.mode == 'test'
            img = sample['input']
            img = np.transpose(img, axes=[3, 0, 1, 2])
            img = torch.from_numpy(img).float()
            sample['input'] = img

        return sample
    
class ToTensor1:  
    def __call__(self, sample):
        img, mask = sample['input'], sample['target']
        img = np.transpose(img, axes=[3, 0, 1, 2])
        mask = np.transpose(mask, axes=[3, 0, 1, 2])
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()
        sample['input'], sample['target'] = img, mask
        return sample


class Mirroring:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            img, mask = sample['input'], sample['target']

            n_axes = random.randint(0, 3)
            random_axes = random.sample(range(3), n_axes)

            img = np.flip(img, axis=tuple(random_axes))
            mask = np.flip(mask, axis=tuple(random_axes))

            sample['input'], sample['target'] = img.copy(), mask.copy()

        return sample


class NormalizeIntensity:

    def __call__(self, sample):
        img = sample['input']
        img[:, :, :, 0] = self.normalize_ct(img[:, :, :, 0])
        img[:, :, :, 1] = self.normalize_pt(img[:, :, :, 1])

        sample['input'] = img
        return sample

    @staticmethod
    def normalize_ct(img):
        norm_img = np.clip(img, -1024, 1024) / 1024
        return norm_img

    @staticmethod
    def normalize_pt(img):
        mean = np.mean(img)
        std = np.std(img)
        return (img - mean) / (std + 1e-3)


class RandomRotation:
    def __init__(self, p=0.5, angle_range=[5, 15]):
        self.p = p
        self.angle_range = angle_range

    def __call__(self, sample):
        if random.random() < self.p:
            img, mask = sample['input'], sample['target']

            num_of_seqs = img.shape[-1]
            n_axes = random.randint(1, 3)
            random_axes = random.sample([0, 1, 2], n_axes)

            for axis in random_axes:

                angle = random.randrange(*self.angle_range)
                angle = -angle if random.random() < 0.5 else angle

                for i in range(num_of_seqs):
                    img[:, :, :, i] = RandomRotation.rotate_3d_along_axis(img[:, :, :, i], angle, axis, 1)

                mask[:, :, :, 0] = RandomRotation.rotate_3d_along_axis(mask[:, :, :, 0], angle, axis, 0)

            sample['input'], sample['target'] = img, mask
        return sample

    @staticmethod
    def rotate_3d_along_axis(img, angle, axis, order):

        if axis == 0:
            rot_img = rotate(img, angle, order=order, preserve_range=True)

        if axis == 1:
            rot_img = np.transpose(img, axes=(1, 2, 0))
            rot_img = rotate(rot_img, angle, order=order, preserve_range=True)
            rot_img = np.transpose(rot_img, axes=(2, 0, 1))

        if axis == 2:
            rot_img = np.transpose(img, axes=(2, 0, 1))
            rot_img = rotate(rot_img, angle, order=order, preserve_range=True)
            rot_img = np.transpose(rot_img, axes=(1, 2, 0))

        return rot_img


class ZeroPadding:

    def __init__(self, target_shape, mode='train'):
        self.target_shape = np.array(target_shape)  # without channel dimension
        if mode not in ['train', 'test']:
            raise ValueError(f"Argument 'mode' must be 'train' or 'test'. Received {mode}")
        self.mode = mode

    def __call__(self, sample):
        if self.mode == 'train':
            img, mask = sample['input'], sample['target']

            input_shape = np.array(img.shape[:-1])  # last (channel) dimension is ignored
            d_x, d_y, d_z = self.target_shape - input_shape
            d_x, d_y, d_z = int(d_x), int(d_y), int(d_z)

            if not all(i == 0 for i in (d_x, d_y, d_z)):
                positive = [i if i > 0 else 0 for i in (d_x, d_y, d_z)]
                negative = [i if i < 0 else None for i in (d_x, d_y, d_z)]

                # padding for positive values:
                img = np.pad(img, ((0, positive[0]), (0, positive[1]), (0, positive[2]), (0, 0)), 'constant', constant_values=(0, 0))
                mask = np.pad(mask, ((0, positive[0]), (0, positive[1]), (0, positive[2]), (0, 0)), 'constant', constant_values=(0, 0))

                # cropping for negative values:
                img = img[: negative[0], : negative[1], : negative[2], :].copy()
                mask = mask[: negative[0], : negative[1], : negative[2], :].copy()

                assert img.shape[:-1] == mask.shape[:-1], f'Shape mismatch for the image {img.shape[:-1]} and mask {mask.shape[:-1]}'

                sample['input'], sample['target'] = img, mask

            return sample

        else:  # if self.mode == 'test'
            img = sample['input']

            input_shape = np.array(img.shape[:-1])  # last (channel) dimension is ignored
            d_x, d_y, d_z = self.target_shape - input_shape
            d_x, d_y, d_z = int(d_x), int(d_y), int(d_z)

            if not all(i == 0 for i in (d_x, d_y, d_z)):
                positive = [i if i > 0 else 0 for i in (d_x, d_y, d_z)]
                negative = [i if i < 0 else None for i in (d_x, d_y, d_z)]

                # padding for positive values:
                img = np.pad(img, ((0, positive[0]), (0, positive[1]), (0, positive[2]), (0, 0)), 'constant', constant_values=(0, 0))

                # cropping for negative values:
                img = img[: negative[0], : negative[1], : negative[2], :].copy()

                sample['input'] = img

            return sample
        
class ZeroPadding1:

    def __init__(self, target_shape):
        self.target_shape = np.array(target_shape)  # without channel dimension

    def __call__(self, sample):
        img, mask = sample['input'], sample['target']
        input_shape = np.array(img.shape[:-1])  # last (channel) dimension is ignored
        d_x, d_y, d_z = self.target_shape - input_shape
        d_x, d_y, d_z = int(d_x), int(d_y), int(d_z)

        if not all(i == 0 for i in (d_x, d_y, d_z)):
            positive = [i if i > 0 else 0 for i in (d_x, d_y, d_z)]
            negative = [i if i < 0 else None for i in (d_x, d_y, d_z)]

            # padding for positive values:
            img = np.pad(img, ((0, positive[0]), (0, positive[1]), (0, positive[2]), (0, 0)), 'constant', constant_values=(0, 0))
            mask = np.pad(mask, ((0, positive[0]), (0, positive[1]), (0, positive[2]), (0, 0)), 'constant', constant_values=(0, 0))

            # cropping for negative values:
            img = img[: negative[0], : negative[1], : negative[2], :].copy()
            mask = mask[: negative[0], : negative[1], : negative[2], :].copy()

            assert img.shape[:-1] == mask.shape[:-1], f'Shape mismatch for the image {img.shape[:-1]} and mask {mask.shape[:-1]}'

            sample['input'], sample['target'] = img, mask

        return sample




class ExtractPatch:
    """Extracts a patch of a given size from an image (4D numpy array)."""

    def __init__(self, patch_size, p_tumor=0.5):
        self.patch_size = patch_size  # without channel dimension!
        self.p_tumor = p_tumor  # probs to extract a patch with a tumor

    def __call__(self, sample):
        img = sample['input']
        mask = sample['target']

        assert all(x <= y for x, y in zip(self.patch_size, img.shape[:-1])), \
            f"Cannot extract the patch with the shape {self.patch_size} from  " \
                f"the image with the shape {img.shape}."

        # patch_size components:
        ps_x, ps_y, ps_z = self.patch_size

        if random.random() < self.p_tumor:
            # coordinates of the tumor's center:
            xs, ys, zs, _ = np.where(mask != 0)
            tumor_center_x = np.min(xs) + (np.max(xs) - np.min(xs)) // 2
            tumor_center_y = np.min(ys) + (np.max(ys) - np.min(ys)) // 2
            tumor_center_z = np.min(zs) + (np.max(zs) - np.min(zs)) // 2

            # compute the origin of the patch:
            patch_org_x = random.randint(tumor_center_x - ps_x, tumor_center_x)
            patch_org_x = np.clip(patch_org_x, 0, img.shape[0] - ps_x)

            patch_org_y = random.randint(tumor_center_y - ps_y, tumor_center_y)
            patch_org_y = np.clip(patch_org_y, 0, img.shape[1] - ps_y)

            patch_org_z = random.randint(tumor_center_z - ps_z, tumor_center_z)
            patch_org_z = np.clip(patch_org_z, 0, img.shape[2] - ps_z)
        else:
            patch_org_x = random.randint(0, img.shape[0] - ps_x)
            patch_org_y = random.randint(0, img.shape[1] - ps_y)
            patch_org_z = random.randint(0, img.shape[2] - ps_z)

        # extract the patch:
        patch_img = img[patch_org_x: patch_org_x + ps_x,
                    patch_org_y: patch_org_y + ps_y,
                    patch_org_z: patch_org_z + ps_z,
                    :].copy()

        patch_mask = mask[patch_org_x: patch_org_x + ps_x,
                     patch_org_y: patch_org_y + ps_y,
                     patch_org_z: patch_org_z + ps_z,
                     :].copy()

        assert patch_img.shape[:-1] == self.patch_size, \
            f"Shape mismatch for the patch with the shape {patch_img.shape[:-1]}, " \
                f"whereas the required shape is {self.patch_size}."

        sample['input'] = patch_img
        sample['target'] = patch_mask

        return sample


class InverseToTensor:
    def __call__(self, sample):
        output = sample['output']

        output = torch.squeeze(output)  # squeeze the batch and channel dimensions
        output = output.numpy()

        sample['output'] = output
        return sample


class CheckOutputShape:
    def __init__(self, shape=(144, 144, 144)):
        self.shape = shape

    def __call__(self, sample):
        output = sample['output']
        assert output.shape == self.shape, \
            f'Received wrong output shape. Must be {self.shape}, but received {output.shape}.'
        return sample


class ProbsToLabels:
    def __call__(self, sample):
        output = sample['output']
        output = (output > 0.5).astype(int)  # get binary label
        sample['output'] = output
        return sample


class headneckDataset(Dataset):
    
    def __init__(self,imgpath,csvpath,fold,transform=None):
        self.imgpath=imgpath
        self.csvpath=csvpath
        self.transform=transform
        self.fold=fold
        input_folder=Path(self.imgpath)
        pathfile=os.path.join(self.csvpath,self.fold)
        patientfile=pd.read_csv(pathfile)
        self.pathctlist=[]
        self.pathptlist=[]
        self.pathgtvtlist=[]
        self.path_id=[]
        for p in patientfile['PatientID']:
            pathct=str([f for f in input_folder.rglob(p + "_ct*")][0].resolve())
            pathpt=str([f for f in input_folder.rglob(p + "_pt*")][0].resolve())
            pathgtvt=str([f for f in input_folder.rglob(p + "_gtvt*")][0].resolve())
            self.pathctlist.append(pathct)
            self.pathptlist.append(pathpt)
            self.pathgtvtlist.append(pathgtvt)
            self.path_id.append(p)
            #print(pathct)
    
    def __len__(self):
        return(len(self.pathptlist))
    
    def __getitem__(self, idx):
        sample=dict()
        
        pathct=self.pathctlist[idx]
        ct_img=self.read_nifti(pathct,True)
        pathpt=self.pathptlist[idx]
        pt_img=self.read_nifti(pathpt,True)
        imgc=[ct_img,pt_img]
        img = np.stack(imgc, axis=-1)
        
        sample['input']= img
        
        pathmask=self.pathgtvtlist[idx]
        mask=self.read_nifti(pathmask,True)
        mask = np.expand_dims(mask, axis=3)
        sample['target']= mask
        
        sample['id'] = self.path_id[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample
        
    
    def read_nifti(self,path_to_nifti, return_numpy=True):
        """Read a NIfTI image. Return a numpy array (default) or `nibabel.nifti1.Nifti1Image` object"""
        if return_numpy:
            return nib.load(str(path_to_nifti)).get_fdata()
        return nib.load(str(path_to_nifti))
 
pathpatient="/raid/Home/Users/aqayyum/EZProj/Hector2021/hecktor_nii_resampled/"
#input_folder=Path(pathpatient)
pathcsv="/raid/Home/Users/aqayyum/EZProj/Hector2021/CV_ids/"
foldtrain='train_fold0.csv'
foldvalid='valid_fold0.csv'
#pathfile=os.path.join(pathcsv,'train_fold0.csv')    
#dataset=headneckDataset(imgpath=pathpatient,csvpath=pathcsv,fold=fold1,transform=None)       

# print(len(dataset)) 
   
 
# sample=dataset[0]   
# print(sample['id'])  
# print(sample['input'].shape) 
# print(sample['target'].shape)   

#Data transforms
train_transforms = Compose([
    #RandomRotation(p=0.5, angle_range=[0, 45]),
    #Mirroring(p=0.5),
    NormalizeIntensity(),
    ToTensor1()
])



val_transforms =Compose([
    NormalizeIntensity(),
    ToTensor1()
])

train_set = headneckDataset(imgpath=pathpatient,csvpath=pathcsv,fold=foldtrain,transform=train_transforms)
print(len(train_set)) 
val_set = headneckDataset(imgpath=pathpatient,csvpath=pathcsv,fold=foldvalid,transform=val_transforms)
print(len(val_set)) 
   
 
sample=train_set[0]   
print(sample['id'])  
print(sample['input'].shape) 
print(sample['target'].shape)   

sample=val_set[0]   
print(sample['id'])  
print(sample['input'].shape) 
print(sample['target'].shape)  

train_batch_size = 2
val_batch_size = 2
num_workers = 2  # for example, use a number of CPU cores

# # Datasets:
# train_set = dataset.HecktorDataset(train_paths, transforms=train_transforms)
# val_set = dataset.HecktorDataset(val_paths, transforms=val_transforms)
from torch.utils.data import DataLoader
# Dataloaders:
train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=False, num_workers=2)
#from torch.utils.data import DataLoader
dataloader = {
    'train': train_loader,
    'val': val_loader
}

#One training sample
train_sample = next(iter(train_loader))
train_sample.keys()

print(f'Patient: \t{train_sample["id"]}')
print(f'Input: \t\t{train_sample["input"].size()}')
print(f'Target: \t{train_sample["target"].size()}')

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
inp=torch.rand(1,2,144,144,144)
out=model(inp)

# training and validation functions
import os
import pathlib
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

import os
import pathlib
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch


import torch
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1

    def forward(self, input, target):
        axes = tuple(range(1, input.dim()))
        intersect = (input * target).sum(dim=axes)
        union = torch.pow(input, 2).sum(dim=axes) + torch.pow(target, 2).sum(dim=axes)
        loss = 1 - (2 * intersect + self.smooth) / (union + self.smooth)
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = 1e-3

    def forward(self, input, target):
        input = input.clamp(self.eps, 1 - self.eps)
        loss = - (target * torch.pow((1 - input), self.gamma) * torch.log(input) +
                  (1 - target) * torch.pow(input, self.gamma) * torch.log(1 - input))
        return loss.mean()


class Dice_and_FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(Dice_and_FocalLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(gamma)

    def forward(self, input, target):
        loss = self.dice_loss(input, target) + self.focal_loss(input, target)
        return loss
    
###################### metrics #######################
def dice(input, target):
    axes = tuple(range(1, input.dim()))
    bin_input = (input > 0.5).float()

    intersect = (bin_input * target).sum(dim=axes)
    union = bin_input.sum(dim=axes) + target.sum(dim=axes)
    score = 2 * intersect / (union + 1e-3)

    return score.mean()
    
#def IoU(pr, gt, th=0.5, eps=1e-7):
#    pr = torch.sigmoid(pr) > th
#    gt = gt > th
#    intersection = torch.sum(gt * pr, axis=(-2,-1))
#    union = torch.sum(gt, axis=(-2,-1)) + torch.sum(pr, axis=(-2,-1)) - intersection + eps
#    ious = (intersection + eps) / union
#    return torch.mean(ious).item()   
    
def IoU1(input, target, th=0.5, eps=1e-7):
    axes = tuple(range(1, input.dim()))
    bin_input = (input > 0.5).float()

    intersect = (bin_input * target).sum(dim=axes)
    union = bin_input.sum(dim=axes) + target.sum(dim=axes)- intersect + eps
    ious = (intersect + eps) / union
    return torch.mean(ious).item()      
    
  
def recall(input, target):
    axes = tuple(range(1, input.dim()))
    binary_input = (input > 0.5).float()

    true_positives = (binary_input * target).sum(dim=axes)
    all_positives = target.sum(dim=axes)
    recall = true_positives / all_positives

    return recall.mean()


def precision(input, target):
    axes = tuple(range(1, input.dim()))
    binary_input = (input > 0.5).float()

    true_positives = (binary_input * target).sum(dim=axes)
    all_positive_calls = binary_input.sum(dim=axes)
    precision = true_positives / all_positive_calls

    return precision.mean()
# train settings:
train_batch_size: 2
val_batch_size: 2
num_workers: 2  # for example, use a number of CPU cores

lr: 1e-3  # initial learning rate
n_epochs: 300  # number of training epochs (300 was used in the paper)
n_cls: 2  # number of classes to predict (background and tumor)
in_channels: 2  # number of input modalities
n_filters: 24  # number of filters after the input (24 was used in the paper)
reduction: 2  # parameter controls the size of the bottleneck in SENorm layers

T_0: 25  # parameter for 'torch.optim.lr_scheduler.CosineAnnealingWarmRestarts'
eta_min: 1e-5  # parameter for 'torch.optim.lr_scheduler.CosineAnnealingWarmRestarts'
save_path_weights="/raid/Home/Users/aqayyum/EZProj/Hector2021/outputweights/"
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
#model = FastSmoothSENormDeepUNet_supervision_skip_no_drop(in_channels, n_cls, n_filters, reduction)

criterion = Dice_and_FocalLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))
#def dice(input, target):
#    axes = tuple(range(1, input.dim()))
#    bin_input = (input > 0.5).float()
#
#    intersect = (bin_input * target).sum(dim=axes)
#    union = bin_input.sum(dim=axes) + target.sum(dim=axes)
#    score = 2 * intersect / (union + 1e-3)
#
#    return score.mean()
#metric = dice
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, eta_min=1e-5)
import os
import numpy as np
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
# modelUdense=ResUNet(n_channels, n_classes)
# print(modelUdense)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#lr=3e-4
criterion = Dice_and_FocalLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))
batch_size=2  
alpha = 0.4 
from tqdm import tqdm
from collections import OrderedDict
def dice(input, target):
    axes = tuple(range(1, input.dim()))
    bin_input = (input > 0.5).float()

    intersect = (bin_input * target).sum(dim=axes)
    union = bin_input.sum(dim=axes) + target.sum(dim=axes)
    score = 2 * intersect / (union + 1e-3)

    return torch.mean(score).item()  
    
    
def IoU(input, target, th=0.5, eps=1e-7):
    axes = tuple(range(1, input.dim()))
    bin_input = (input > 0.5).float()

    intersect = (bin_input * target).sum(dim=axes)
    union = bin_input.sum(dim=axes) + target.sum(dim=axes)- intersect + eps
    ious = (intersect + eps) / union
    return torch.mean(ious).item()     

##################### training function ##########
##################### training function ##########
def train(dataloader, model, criterion, optimizer, epoch, scheduler=None):
    bar = tqdm(dataloader['train'])
    losses_avg, ious_avg = [], []
    train_loss, train_iou = [], []
    model.to(device)
    model.train()
    for sample in bar:
        imgs, masks = sample['input'], sample['target']
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        y_hat = model(imgs)     
#        loss0 = criterion(y_hat[0], masks)
#        loss1 = criterion(y_hat[1], masks)
#        loss2 = criterion(y_hat[2], masks)
#        loss3 = criterion(y_hat[3], masks)
#
#        loss = loss3  +  alpha * (loss0 + loss1 + loss2)
#        loss.backward()
#        optimizer.step()
        
        #train_loss.update(loss3.item(),data.size(0))
        #ious=dice(y_hat, masks)
#        train_loss.append(loss.item())
#        ious=dice(y_hat[3], masks)
#        train_iou.append(ious)      
        loss = criterion(y_hat, masks)
        loss.backward()
        optimizer.step()
        #ious = IoU(y_hat, masks)
        ious=dice(y_hat, masks)
        train_loss.append(loss.item())
        train_iou.append(ious)
        #bar.set_description(f"loss {np.mean(train_loss):.5f} iou {np.mean(train_iou):.5f}")
    losses_avg=np.mean(train_loss)
    ious_avg=np.mean(train_iou)
    
    log = OrderedDict([('loss', losses_avg),
                       ('iou', ious_avg),
                       ])
    return log

def validate(dataloader, model, criterion):
    bar = tqdm(dataloader['val'])
    test_loss, test_iou = [], []
    losses_avg, ious_avg = [], []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for sample in bar:
            imgs, masks = sample['input'], sample['target']
            imgs, masks = imgs.to(device), masks.to(device)
            y_hat = model(imgs)
            loss = criterion(y_hat, masks)
            #ious = IoU(y_hat, masks)
            ious=dice(y_hat, masks)
            test_loss.append(loss.item())
            test_iou.append(ious)
            bar.set_description(f"test_loss {np.mean(test_loss):.5f} test_iou {np.mean(test_iou):.5f}")
    losses_avg=np.mean(test_loss)
    ious_avg=np.mean(test_iou)
    log = OrderedDict([('loss', losses_avg),
                       ('iou', ious_avg),
                       ])
    
    return log 
import pandas as pd
lr=1e-3
#criterion = torch.nn.BCEWithLogitsLoss()
log = pd.DataFrame(index=[], columns=['epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'])
early_stop=20
epochs=10000
best_iou = 0
name='3DDeepinceptionRes_fold0'
trigger = 0
for epoch in range(epochs):
    print('Epoch [%d/%d]' %(epoch, epochs))
    # train for one epoch
    train_log = train(dataloader, model, criterion, optimizer, epoch)
    #train_log = train(train_loader, model, optimizer, epoch)
    # evaluate on validation set
    #val_log = validate(val_loader, model)
    val_log =validate(dataloader, model, criterion)
    print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'%(train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

    tmp = pd.Series([epoch,lr,train_log['loss'],train_log['iou'],val_log['loss'],val_log['iou']], index=['epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'])

    log = log.append(tmp, ignore_index=True)
    log.to_csv('models/%s/log.csv' %name, index=False)

    trigger += 1

    if val_log['iou'] > best_iou:
        torch.save(model.state_dict(), 'models/%s/3DDeepinceptionRes_fold0.pth' %name)
        best_iou = val_log['iou']
        print("=> saved best model")
        trigger = 0

    # early stopping
    if not early_stop is None:
        if trigger >= early_stop:
            print("=> early stopping")
            break

    torch.cuda.empty_cache()
print("done training")