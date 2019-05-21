# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 13:25:58 2018

@author: kxshi
"""
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits 
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.neighbors import KernelDensity as KD
from sklearn.model_selection import GridSearchCV as GSCV, cross_val_score
from sklearn.base import BaseEstimator as BE, ClassifierMixin as CM

sns.set()
def make_data(N, f=0.3, seed=1):
    rng = npr.RandomState(seed)
    x = rng.randn(N)
    x[int(f*N):] += 5
    return x

x = make_data(1000)
x_new = np.linspace(-4, 8, 1000)
dig = load_digits()

'''
plt.figure(figsize=(12,8))
hist = plt.hist(x, bins=30, density=True)
density, bins, patchs= hist
print(density.shape,bins.shape) # (30,) (31,)
area = np.dot((bins[1:]-bins[:-1]), density)
print((bins[-1]-bins[0])/30, bins[1]-bins[0], area, sep='\t')

# KD model
kde = KD(bandwidth=1.0, kernel='gaussian')
kde.fit(x[:, None])  # 特征矩阵
logprob = kde.score_samples(x_new[:, np.newaxis]) #返回概率密度的对数值
plt.fill_between(x_new, np.exp(logprob), alpha=0.6) # 标准化处理，area=1.0
plt.plot(x, np.full_like(x, -0.01), '|r', mew=1)
plt.xlim(-4, 8)
plt.ylim(-0.02, 0.22)
print(np.dot(np.full(x_new.shape,x_new[1]-x_new[0]), np.exp(logprob)))

# 网格搜索，找最优参数bandwidth, 核大小
bandwidths = np.linspace(-2, 2, 50)
grid = GSCV(KD(kernel='gaussian'), {'bandwidth':bandwidths}, 
            cv=200).fit(x[:, None])
print(grid.best_params_)
#  GSCV(estimator, param_grid, scoring=None,  cv=None,
'''

# 自定义评估器estimator  一个类， 继承BaseEstimator类,混合类ClassifierMixin
class KDEClassifier(BE, CM):
    """基于KDE的贝叶斯生成分类
    参数
    -------
    bandwidth: float
        每个类中的核带宽，表示核的大小
    kernel: str
        核函数(6个)名称， 传递给KernelDensity
    -------
    """
    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        # 初始化，将参数值传递给属性， 不使用*args和**kwargs
        self.bandwidth = bandwidth
        self.kernel = kernel
        
    def fit(self, x, y):
        # 处理训练数据
        self.classes_ = np.sort(np.unique(y)) 
        training_sets = [x[y==yi] for yi in self.classes_ ] # 标签去重后分类
        self.models_ = [KD(bandwidth=self.bandwidth,kernel=self.kernel).fit(xi)
                    for xi in training_sets] # 每类训练一个KD模型
        self.logpriors_ = [np.log(xi.shape[0]/x.shape[0])  # 计算先验概率
                    for xi in training_sets]
        return self
    
    def predict_proba(self, x):
        # 预测新数据   返回每一类的概率数组[n_samples, n_classes]
        logprobs = np.vstack([model.score_samples(x) 
                    for model in self.models_]).T
        result = np.exp(logprobs + self.logpriors_)
        return result / result.sum(axis=1, keepdims=True)
    
    def predict(self, x):
        # 概率分类器
        return self.classes_[np.argmax(self.predict_proba(x), axis=1)]
    

# 使用自定义的评估类
bandwidths = 10 ** np.linspace(0, 2, 100)
grid = GSCV(KDEClassifier(), {'bandwidth': bandwidths}).fit(dig.data, dig.target)
scores = [val.mean_validation_score for val in grid.grid_scores_]

# 交叉检验值分数曲线
plt.figure(figsize=(12, 8))
plt.semilogx(bandwidths, scores)
plt.xlabel('bandwidth')
plt.ylabel('accuracy')
plt.title('KDE Model Performance')
print(grid.best_params_, grid.best_score_)
print(cross_val_score(GNB(), dig.data, dig.target).mean())  

