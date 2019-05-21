# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 10:37:54 2018

@author: kxshi
"""

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import mode
from sklearn.datasets import make_blobs, make_moons, load_digits, load_sample_image
from sklearn.cluster import KMeans, SpectralClustering as SC, MiniBatchKMeans as MBK
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, pairwise_distances_argmin as p_d_a

sns.set(style='whitegrid')

x, y = make_blobs(n_samples=300, centers=4, random_state=0, cluster_std=0.6) 
x1, y1 = make_moons(200, noise=0.05, random_state=0)
digits = load_digits() # digits.data  1797x64
china = load_sample_image('china.jpg')


'''
model = KMeans(n_clusters=4).fit(x)  # 300x2
y_pred = model.predict(x)
centers = model.cluster_centers_
print(centers) # 簇中心坐标
plt.figure()
plt.scatter(x[:,0], x[:,1], s=50, c=y_pred, cmap='viridis')
plt.scatter(centers[:,0], centers[:,1], c='r', s=150, alpha=0.6)


y_pred1 = KMeans(2, random_state=0).fit_predict(x1)  # 线性边界
model1 = SC(2, affinity='nearest_neighbors', assign_labels='kmeans')
y_pred2 = model1.fit_predict(x1) # 非线性边界
fig, ax = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
ax[0].scatter(x1[:,0], x1[:,1], s=50, c=y_pred1, cmap='rainbow') # 线性边界
ax[1].scatter(x1[:,0], x1[:,1], s=50, c=y_pred2, cmap='rainbow') # 非线性边界


# EM算法 Expectation Maximization期望最大值
# E-step 点分配至离其最近的簇中心点 / M-step 簇中心点设置为所有点坐标的平均值Mean
def find_clusters(x, n_clusters, seed=2):
    rng = npr.RandomState(seed)
    i = rng.permutation(x.shape[0])[:n_clusters] # Permuted sequence  序列改变
    print(i) # [ 98 259 184 256] 重新排列
    centers= x[i]
    print(centers) # 初始随即设置中心点
    
    while True:
        # 1基于当前最近的中心指定标签
        labels = p_d_a(x, centers) 
        # 2根据点的平均值找到新的中心
        print([x[labels==i].shape for i in range(n_clusters)]) 
        # [(66, 2), (87, 2), (42, 2), (41, 2), (64, 2)]
        new_centers = np.array([x[labels==i].mean(0) for i in range(n_clusters)])
        # 3确认收敛
        if np.all(centers == new_centers):
            break
        centers = new_centers
        
    return centers, labels

centers, labels = find_clusters(x, 5)
print(centers)
print(labels)
plt.figure(figsize=(8, 8))
plt.scatter(x[:,0], x[:,1], c=labels, cmap='viridis', s=50)
'''

# 处理手写数字
'''
model = KMeans(n_clusters=10, random_state=0)
labels = model.fit_predict(digits.data) # (1797, 64)
print(labels.shape) # 1x1797
centers = model.cluster_centers_.reshape(10, 8, 8)  # 10x64 ——10x8x8
fig, ax = plt.subplots(2, 5, figsize=(10, 4))
for i, center, axi in zip(range(len(centers)), centers, ax.flat):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest', cmap='binary')

# 预处理降维  优化
tsne = TSNE(n_components=2, init='pca',random_state=0) 
pro = tsne.fit_transform(digits.data)
print(pro.shape) # (1797, 2)

model1 = KMeans(n_clusters=10, random_state=0)
labels = model1.fit_predict(pro) # (1797, 2) /(1, 1797)
    
# 排列标签
label = np.zeros_like(labels)
for i in range(10):
    mask = (labels == i) # 掩码数组 1x1797 对应的为True,不对应的为False
    label[mask] = mode(digits.target[mask])[0]
    
print(accuracy_score(digits.target, label))
'''

# 色彩压缩
ax = plt.axes(xticks=[], yticks=[]) # 使用axes呈现图片，去刻度
ax.imshow(china)

data = china / 255.0  # 转换为0-1区间
data = data.reshape(427*640, 3)
print(china.shape, data.shape) # (427, 640, 3) 三维颜色空间  (2732800, 3)
'''
def plot_pixels(data, colors=None, N=10000):
    # 像素可视化
    if colors is None:
        colors = data # (2732800, 3)
    rng = npr.RandomState(0)
    i = rng.permutation(data.shape[0])[:N] # 取前10000个
    colors = colors[i] # 10000x3
    R, G, B = data[i].T  # 3RGB  x 10000   # (10000,) (10000,) (10000,)
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    ax[0].scatter(R, G, color=colors, marker='.')
    ax[0].set(xlabel='red', ylabel='green', xlim=(0,1), ylim=(0, 1))
    ax[1].scatter(R, B, color=colors, marker='.')
    ax[1].set(xlabel='red', ylabel='blue', xlim=(0,1), ylim=(0, 1))
    
plot_pixels(data)
'''

             
fig, ax = plt.subplots(2, 2, figsize=(12,8),
                   subplot_kw=dict(xticks=[], yticks=[]))
fig.subplots_adjust(wspace=0.05)
for i, axi in enumerate(ax.flat):
    model3 = MBK(n_clusters=16*(2*i+1)).fit(data)  # 拟合数据，分为16*i类(缩减到16个颜色)
    labels = model3.predict(data) #  shape (273280,)
    new_c = model3.cluster_centers_[labels]  # 16个簇中心点坐标  shape(273280, 3)
    # plot_pixels(data, colors=new_c)  # 新颜色方案
    new_china = new_c.reshape(china.shape) # 恢复到原shape
    axi.imshow(new_china)
    axi.set_title('{0}-color Image'.format(16*(2*i+1)), size=16) 
   