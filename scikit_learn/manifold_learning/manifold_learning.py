# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 19:26:50 2018

@author: kxshi
"""

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_mldata, fetch_lfw_people as f_l_p, load_digits
from sklearn.metrics import pairwise_distances as p_d
from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding as LLE
from sklearn.decomposition import PCA
sns.set()

faces = f_l_p(min_faces_per_person=60)
digits = load_digits()
data = fetch_mldata('Whistler Daily Snowfall')
print(faces.data.shape, digits.data.shape, data.data.shape) 
# 1348x2914  pixal   2914维     (1797, 64) (13880, 9)

'''
def make_hello(N=1000, seed=42):
    fig, ax = plt.subplots(figsize=(4, 1))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    ax.text(0.5, 0.4, 'HELLO', va='center', ha='center',weight='bold', size=85)
    fig.savefig('hello.png')
    plt.close(fig)
    
    # 打开PNG,将一些随机点画进去
    data = imread('hello.png')[::-1, :, 0].T
    rng = npr.RandomState(seed)
    x = rng.rand(4*N, 2)
    i,j = (x*data.shape).astype(int).T
    x = x[(data[i,j] < 1)]
    print(data.shape, x.shape)  # (288, 72) (1532, 2)
    x[:, 0] *= (data.shape[0]/data.shape[1])
    x = x[:N]
    return x[np.argsort(x[:, 0])]

x = make_hello(1000)
plt.scatter(x[:, 0], x[:, 1], c=x[:, 0], cmap=plt.cm.get_cmap('rainbow',5))
plt.axis('equal')

D = p_d(x) # 数据点的距离矩阵
print(x.shape, D.shape) # (1000,2)  /  (1000, 1000)
print(x[:3])
print(D[:3, :3]) # 实对称矩阵，对角线都为0
plt.figure(figsize=(8,8))
plt.imshow(D, zorder=2, cmap='Blues', interpolation='nearest')

model = MDS(random_state=1, dissimilarity='precomputed')
out = model.fit_transform(D)
plt.figure()
plt.scatter(out[:,0], out[:,1],c=x[:, 0],cmap=plt.cm.get_cmap('rainbow',5))
plt.axis('equal')

model1 = LLE(n_neighbors=100, n_components=2, method='modified', 
             eigen_solver='dense')
out1 = model.fit_transform(D)
plt.figure()
plt.scatter(out1[:,0], out1[:,1],c=x[:, 0],cmap=plt.cm.get_cmap('rainbow',5))
plt.axis('equal')

fig, axes = plt.subplots(4, 8, figsize=(12,8),
                         subplot_kw=dict(xticks=[], yticks=[]))
for i, ax in enumerate(axes.flat):
    ax.imshow(faces.data[i].reshape(62, 47), cmap='gray')
'''

model2 = PCA(n_components=100, svd_solver='randomized').fit(faces.data)
plt.figure()
plt.plot(np.cumsum(model2.explained_variance_ratio_))

model3 = Isomap()
projection = model3.fit_transform(faces.data)
pro = model3.fit_transform(digits.data[::5]) # 选取部分数据
print(projection.shape, pro.shape) # (1348, 2) (360, 2)
plt.figure()
plt.scatter(pro[:,0], pro[:,1], c=digits.target[::5],
            cmap=plt.cm.get_cmap('jet', 10))
plt.colorbar(ticks=range(10), extend='both')
plt.clim(-0.5, 9.5)

print(dir(fetch_mldata))