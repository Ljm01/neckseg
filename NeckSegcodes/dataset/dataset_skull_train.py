
from torch.utils.data import DataLoader
import os
import sys
import torch
from torch.utils.data import Dataset as dataset
import nibabel as nib
import numpy as np
import SimpleITK as sitk
try:
    from scipy.special import comb
except:
    from scipy.misc import comb

def to1(img):
    min_val = img.min()
    max_val = img.max()
    img = (img - min_val) / (max_val - min_val + 1e-5)  # 图像归一化
    return img


class Train_Dataset(dataset):
    def __init__(self, images_names):


        self.images_list = images_names

    def __getitem__(self, index):

        ct_array = sitk.GetArrayFromImage(sitk.ReadImage(str(self.images_list[index])))
        seg_array = sitk.GetArrayFromImage(sitk.ReadImage(str(self.images_list[index]).replace("image-", "label-").replace("-cut", "-sparse-cut")))

        ct_array = to1(ct_array)
        seg_array[seg_array == 8] = 9
        seg_array[seg_array == 7] = 8
        seg_array[seg_array == 6] = 7
        seg_array[seg_array == 5] = 6
        seg_array[seg_array == 4] = 5
        seg_array[seg_array == 3] = 4
        seg_array[seg_array == 2] = 3
        seg_array[seg_array == 1] = 2
        seg_array[seg_array == -1] = 1

        ct_array_resize = ct_array.astype(np.float32)
        ct_array_resize = torch.FloatTensor(ct_array_resize).unsqueeze(0)
        seg_array_resize = torch.FloatTensor(seg_array)

        return ct_array_resize, seg_array_resize

    def __len__(self):
        return len(self.images_list)
