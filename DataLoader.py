# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 19:40:26 2021

@author: Abdul Qayyum
"""

#%% Hector2021 challenege dataloader prepartion
####################### dataset######################
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