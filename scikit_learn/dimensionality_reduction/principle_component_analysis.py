# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 13:43:52 2018

@author: kxshi
"""

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.datasets
import sklearn.model_selection, sklearn.ensemble
from sklearn.datasets import load_digits, fetch_lfw_people as f_l_p
from sklearn.decomposition import PCA


sns.set()

rng = npr.RandomState(0)
x = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T # 200x2 特征矩阵
digits = load_digits() # 1797x64(8x8像素的图像)
faces = f_l_p(min_faces_per_person=60) 
print(faces.images.shape, faces.data.shape)# 1348x62x47 / 1348x2914
pca = PCA().fit(x) # n_components=2, random_state=None  default

print(pca.components_, pca.explained_variance_)
# [[ 0.739393    0.67327408] [-0.67327408  0.739393  ]]  2x2
# [1.48668691 0.01057775]  

'''
def draw(v0, v1):
    ax = plt.gca()
    arrow = dict(arrowstyle='->',color='r',linewidth=2,shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrow)
    
plt.figure(figsize=(12, 8))
plt.scatter(x[:,0], x[:,1], alpha=0.3)
for len, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(len)
    print(v) # [0.9015402  0.82092154] // [-0.06924501  0.07604523]
    draw(pca.mean_, pca.mean_+v)
plt.axis('equal')

# 降维和逆变换
pca1 = PCA(n_components=1).fit(x)
x_pca = pca1.transform(x)
print(x.shape, x_pca.shape) # (200, 2) (200, 1)
x_new = pca1.inverse_transform(x_pca) # 降维逆变换 (200, 2)
plt.figure(figsize=(8, 8))
plt.scatter(x[:, 0], x[:, 1], alpha=0.3) # 原始数据
plt.scatter(x_new[:, 0], x_new[:, 1], alpha=0.7) 
# 逆变换后的数据， 仅留下含有最高方差值的数据成分 
plt.axis([-3, 3, -3, 3])

# 手写数字 从64维投影至2维, 模型训练前进行降维处理  可视化
pca2 = PCA()
projected = pca2.fit_transform(digits.data)  
# 1797x64 ——> 1797x2  每个数据点沿最大方差方向的投影
plt.figure(figsize=(12,8))
plt.scatter(projected[:,0], projected[:,1], c=digits.target,
            cmap=plt.cm.get_cmap('Spectral', 10), alpha=0.6)
plt.colorbar()

# 选择成分的数量, 以便正确描述数据  累计方差贡献率——成分数量 函数
# 需要20个成分来 保持90%的方差 (according to curve)
pca3 = PCA().fit(digits.data)
res = np.cumsum(pca3.explained_variance_ratio_)
plt.figure(figsize=(8, 8))
plt.plot(np.arange(64), res)
plt.xlabel('num of components')
plt.ylabel('explained variance ratio')
plt.axis([0, 63, res[0], res[-1]])

# 噪音过滤  手写数字
def plot_digits(data):
    fig, ax = plt.subplots(4, 10, figsize=(10, 8),
                           subplot_kw=dict(xticks=[], yticks=[]),
                           gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i , ax in enumerate(ax.flat):
        ax.imshow(data[i].reshape(8, 8), cmap='binary', 
                  interpolation='nearest', clim=(0, 16))

plot_digits(digits.data) # 1797x64
noisy = npr.normal(digits.data, 4)  # 带噪音的数据
# normal(loc=0.0, scale=1.0, size=None) 
# Draw random samples from a normal (Gaussian) distribution
print(noisy.shape) # (1797, 64)
plot_digits(noisy)

pca4 = PCA(0.5).fit(noisy) # 投影后保存50%的方差 
# select the number of components such that the amount of variance 
# that needs to be explained
print(pca4.n_components_) # 对应的主成分个数为 n_components_=12
components = pca4.transform(noisy) # 降维  1797x12
fliter = pca4.inverse_transform(components)  # 降维逆变换  1797x64
print(components.shape, fliter.shape)
plot_digits(fliter)

# 特征脸 案例 
pca5 =  PCA(n_components=150, svd_solver='randomized').fit(faces.data) 
# svd_solver='randomized' 随机方法来估计150个主成分， 快速近似
print(pca5.components_.shape) # (150, 2914)  成分数, 特征数
fig, ax = plt.subplots(4, 10, figsize=(10, 8),
                           subplot_kw=dict(xticks=[], yticks=[]),
                           gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i , ax in enumerate(ax.flat):
    ax.imshow(pca5.components_[i].reshape(62, 47), cmap='bone')

res = np.cumsum(pca5.explained_variance_ratio_)
print('150个成分包含的方差比例：{0:.2%}'.format(res[-1]))
components = pca5.transform(faces.data) #  (1348, 150)
projected = pca5.inverse_transform(components) #  (1348, 2914)
print(components.shape, projected.shape)

fig, ax = plt.subplots(2, 10, figsize=(10, 4),
                           subplot_kw=dict(xticks=[], yticks=[]),
                           gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i in range(10):
    ax[0, i].imshow(faces.data[i].reshape(62, 47), cmap='binary_r')
    ax[1, i].imshow(faces.data[i].reshape(62, 47), cmap='binary_r')
ax[0, 0].set_ylabel('full-dim',fontdict=dict(color='r', size=18))
ax[1, 0].set_ylabel('150-dim', fontdict=dict(color='r', size=18))
'''

print(dir(sklearn.model_selection))



