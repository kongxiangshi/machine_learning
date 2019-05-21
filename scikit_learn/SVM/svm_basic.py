# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns
from sklearn.datasets import make_blobs, make_circles
from sklearn.svm import SVC


sns.set()
# plt.rc('font', family='SimHei')
x, y = make_blobs(n_samples=150, n_features=2, centers=2, random_state=0, cluster_std=0.6)
x1, y1 = make_circles(n_samples=200, factor=0.1, noise=0.1)   # 线性不可分
x2, y2 = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0, cluster_std=1.2)
xfit = np.linspace(-1, 3.5, 300)

def different_margin(x, y, xfit):
    plt.figure(figsize=(12, 10))
    plt.scatter(x[:,0], x[:,1], c=y, s=50, cmap='rainbow')
    plt.plot([0.6], [1.9], 'o', color='g', markersize=13)
    plt.text(0.4, 2.1, '(0.6,1.9)')
    for k , b, width in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
        yfit = k * xfit + b
        plt.plot(xfit, yfit, '-k', alpha=0.7)  # 三个不同的分割器
        plt.fill_between(xfit, yfit-width, yfit+width, edgecolor='none', color='c', alpha=0.3)
        plt.text(-0.5, k*(-0.5)+b, 'y={0}x+{1}'.format(k, b), fontdict=dict(size=20, color='r'))
    plt.axis([-1, 3.5, -1, 6])

def plot_svc(model, a, ax=None, plot_sup=True, text_flag=True):
    """可视化SVC后的数据"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(*ylim, 30)
    X, Y = np.meshgrid(x, y)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    Z = model.decision_function(xy).reshape(X.shape) # (900, 2) / (30, 30)
    # 画出决策边界/等高线(三维空间投影到二维)
    ax.contour(X, Y, Z, colors=['k', 'r', 'k'], levels=[-1, 0, 1], linestyles=['--', '-', '--'])
    # 标记出支持向量, 成功拟合的关键因素
    if plot_sup:
        vec = model.support_vectors_
        ax.scatter(vec[:, 0], vec[:, 1], s=250, facecolor='none', edgecolors='k', linewidths=1)
        if text_flag:
            for i in range(vec.shape[0]):
                ax.text(vec[i, 0]+a, vec[i, 1]-a, '({0:.3f},{1:.3f})'.format(vec[i, 0], vec[i, 1]),
                        fontdict=dict(size=16, color='k'), ha='center', va='center')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.text(0.05, 0.05, '*用圆圈标记的为支持向量*', fontsize=20, transform=ax.transAxes, fontproperties='SimHei')

def linear_plot(x, y):
    model = SVC(kernel='linear', C=1E10, gamma='auto').fit(x, y)
    plt.figure(figsize=(10, 8))
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap='rainbow', s=50)
    plt.title('SVM决策边界模拟效果图', fontdict=dict(size=24, color='r'), fontproperties='SimHei')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis([-1.0, 4, -1, 6])
    plot_svc(model, a=0.3)

def plot3D(x, y):
    """线性不可分"""
    r = np.exp(-(x ** 2).sum(axis=1))
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(projection='3d')
    ax.scatter3D(x[:, 0], x[:, 1], r, s=50, c=y, cmap='rainbow')
    # ax.view_init(elev=30, azim=30)
    ax.set(xlabel='X',ylabel='Y',zlabel='r', title='RBF kernel')

def nonlinear_plot(x, y):
    plt.figure(figsize=(12, 8))
    plt.scatter(x[:, 0], x[:, 1], c=y, s=20, cmap='Spectral')
    model = SVC(kernel='rbf', C=1E6, gamma='auto').fit(x, y)
    plot_svc(model, a=0.1)

def margin_soft(x, y):
    fig, ax = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
    fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1, hspace=0.1)
    ls = [0.1, 1, 10, 100]
    for i, axi in enumerate(ax.flat):  # numpy.flatiter object
        model3 = SVC(kernel='linear', C=ls[i], gamma='auto').fit(x, y)
        # 调节C的值, 软化分类边界线
        axi.scatter(x[:, 0], x[:, 1], s=50, c=y, cmap='jet')
        plot_svc(model3, a=0.01 ,ax=axi, text_flag=False)
        axi.set_title('C={0:.1f}'.format(ls[i]), size=14, color='r')


different_margin(x, y, xfit)
linear_plot(x, y)
plot3D(x1, y1)
nonlinear_plot(x1, y1)
margin_soft(x2, y2)

plt.show()
