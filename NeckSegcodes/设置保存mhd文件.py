import glob

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

names=  glob.glob(r'G:\Neck3DSeg\traing data\image-003-reformed-cut-rz.mhd')

'''

(512, 512, x)

[  0. 205. 420. 500. 550. 600. 820. 850.]
'''
for name in names:
    print(name)
    itkimage = sitk.ReadImage(name)
    sitk.WriteImage(itkimage, name.replace('traing data', 'pred'))

    # print(itkimage)   #这部分给出了关于图像的信息,可以打印处理查看
    # image = sitk.GetArrayFromImage(itkimage)
    #
    #
    # # print(img.shape)
    # print(image.shape)
    # print(np.unique(image))
    # print('*'*20)
