# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier as DTC

x, y = make_blobs(n_samples=300, n_features=2, centers=4, random_state=0, cluster_std=1.0)

def visualize(model, x, y, ax=None, cmap='rainbow'):
    """分类结果可视化"""
    ax = ax or plt.gca()
    ax.scatter(x[:, 0], x[:, 1], c=y, s=30, cmap=cmap, clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')

    model.fit(x, y)
    xx, yy =np.meshgrid(np.linspace(*ax.get_xlim(), 200), np.linspace(*ax.get_ylim(), 200))
    z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)  # (40000,2) -- (40000,) -- (200, 200)
    print(np.unique(z.ravel()))  # [0 1 2 3]
    n_classes = len(np.unique(y))
    ax.contourf(xx, yy, z, alpha=0.3, levels=np.arange(n_classes+1)-0.5, cmap=cmap, zorder=1)
    # 等高线level=[-0.5, 0.5, 1.5, 2.5, 3.5] / 相隔区间(0 , 1, 2, 3)填充不同颜色
    # floating point numbers indicating the level curves to draw, in increasing order
    ax.set(xlim=ax.get_xlim(), ylim=ax.get_ylim())


visualize(DTC(), x, y)

plt.show()