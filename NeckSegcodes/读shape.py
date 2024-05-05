import glob

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

names=  glob.glob(r'D:\bishe\ex\traing data\label-00*-*-sparse-cut-rz.mhd')


for name in names:
    print(name)
    itkimage = sitk.ReadImage(name)
    # print(itkimage)   #这部分给出了关于图像的信息,可以打印处理查看
    image = sitk.GetArrayFromImage(itkimage).transpose(2, 1, 0)


    # print(img.shape)
    print(image.shape)
    print(np.unique(image))
    print('*'*20)
