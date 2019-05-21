# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 20:21:13 2018

@author: kxshi
"""

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_lfw_people as f_l_p
from sklearn.feature_extraction.image import PatchExtractor as PE 
from skimage import data, color, feature, transform
import skimage.data

sns.set()
'''
# 方向梯度直方图 Histogram of Oriented Gradients (HOG) 图像像素转换为向量形式
# 简单的特征提取程序 ， 获取HOG特征
image = color.rgb2gray(data.chelsea())
hog_vec, hog_vis = feature.hog(image, visualise=True)

fig, ax = plt.subplots(1, 2, figsize=(12, 8),
                       subplot_kw=dict(xticks=[], yticks=[]))
ax[0].imshow(image, cmap='gray')
ax[0].set(title='input image')

ax[1].imshow(hog_vis)
ax[1].set_title('visualization of HOG features')
'''
# 简单人脸识别器
faces = f_l_p()
positive_patches = faces.images
print(positive_patches) 

imgs = ['camera', 'text', 'coins', 'moon', 'page', 'clock',
        'immunohistochemistry','chelsea', 'coffee', 'hubble_deep_field']
images = [color.rgb2gray(getattr(data, name)()) for name in imgs]

def extract_patches(img, N, scale=1.0, patch_size=positive_patches[0].shape):
    extracted_patch_size = tuple((scale*np.array(patch_size)).astype(int))
    extractor = PE(patch_size=extracted_patch_size, max_patches=N, 
                   random_state=0)
    patches = extractor.transform(img[np.newaxis])
    if scale != 1:
        patches = np.array([transform.resize(patch, patch_size) 
                    for patch in patches])
    return patches

negative_patches = np.vstack([extract_patches(im, 1000, scale)
            for im in images for scale in [0.5, 1.0, 2.0]])
print(negative_patches.shape)  # 未经识别的图像

fig, ax = plt.subplots(6, 10, figsize=(12, 8))
for i, axi in enumerate(ax.flat):
    axi.imshow(negative_patches[500*i], cmap='gray')
    axi.axis('off')
    

    