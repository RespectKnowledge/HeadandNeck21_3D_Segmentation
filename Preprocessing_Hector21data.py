# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 15:07:54 2021

@author: Abdul Qayyum
"""

#%% Convert dataset into same resampling format
# this is necessary preprocessing step to resample both modality PET/CT

import os
import sys
import pathlib
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt

input_folder="C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\Hector2021\\testing2021hector\\hecktor2021_test\\hecktor2021_test\\hecktor_nii"
input_folder = Path(input_folder)

path_bb = 'C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\Hector2021\\testing2021hector\\hecktor2021_test\\hecktor2021_test\\hecktor2021_bbox_testing.csv'

#patient_list = [f.name.split("_")[0] for f in input_folder.rglob("*_ct*")]
output_folder="C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\Hector2021\\testing2021hector\\hecktor2021_test\\hecktor_nii_resampled"
output_folder = Path(output_folder)
output_folder.mkdir(exist_ok=True)


#print('resampling is {}'.format(str(resampling)))
bb_df = pd.read_csv(path_bb)
bb_df = bb_df.set_index('PatientID')

patient_list = [f.name.split("_")[0] for f in input_folder.rglob("*_ct*")]
resampling=(1,1,1)
print('resampling is {}'.format(str(resampling)))
resampler = sitk.ResampleImageFilter()
resampler.SetOutputDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
resampler.SetOutputSpacing(resampling)

#%matplotlib inline
def resample_one_patient(p):
        bb = np.array([
            bb_df.loc[p, 'x1'], bb_df.loc[p, 'y1'], bb_df.loc[p, 'z1'],
            bb_df.loc[p, 'x2'], bb_df.loc[p, 'y2'], bb_df.loc[p, 'z2']
        ])
        size = np.round((bb[3:] - bb[:3]) / resampling).astype(int)
        ct = sitk.ReadImage(
            str([f for f in input_folder.rglob(p + "_ct*")][0].resolve()))
        pt = sitk.ReadImage(
            str([f for f in input_folder.rglob(p + "_pt*")][0].resolve()))
        # gtvt = sitk.ReadImage(
        #     str([f for f in input_folder.rglob(p + "_gtvt*")][0].resolve()))
        
        resampler.SetOutputOrigin(bb[:3])
        resampler.SetSize([int(k) for k in size])  # sitk is so stupid
        resampler.SetInterpolator(sitk.sitkBSpline)
        ct = resampler.Execute(ct)
        pt = resampler.Execute(pt)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        #gtvt = resampler.Execute(gtvt)
        
        sitk.WriteImage(ct, str((output_folder / (p + "_ct.nii.gz")).resolve()))
        
        sitk.WriteImage(pt, str((output_folder / (p + "_pt.nii.gz")).resolve()))
        
        #sitk.WriteImage(gtvt,str((output_folder / (p + "_gtvt.nii.gz")).resolve()))
# p=patient_list[0]
# resample_one_patient(p)
for p in patient_list:
        resample_one_patient(p)