# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 13:12:35 2018
@author: kxshi
"""
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs, make_moons, load_digits
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture


sns.set()
rng = npr.RandomState(1)
x, y = make_blobs(n_samples=400, n_features=2, centers=4, cluster_std=0.6, random_state=0)
x = x[:, ::-1]  # features交换顺序
x_new = np.dot(x, rng.randn(2, 2))

x1, y1 = make_moons(n_samples=200, noise=0.05, random_state=0)
digits = load_digits()  # digits.data  1797,64


gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=1).fit(x_new)
ypred = gmm.predict(x_new)
prob = gmm.predict_proba(x)
print('计算每个样本归属于某一类的概率:')
print(prob[:5])
print(prob.argmax(axis=1)[:5])

plt.figure(figsize=(12, 8))
size = 100 * prob.max(axis=1)**2   # 每个样本概率最大值的平方
plt.scatter(x[:,0], x[:,1], c=ypred, cmap='rainbow', s=size)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('GaussianMixture', fontdict=dict(size=20, color='r'))

gmm2 = BayesianGaussianMixture(n_components=2, covariance_type='full', max_iter=100, random_state=0)
gmm2.fit(x1)
ypred2 = gmm2.predict(x1)
plt.figure(figsize=(12, 8))
plt.scatter(x1[:,0], x1[:,1], s=50, marker='o', c=ypred2, cmap='Spectral')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('BayesianGaussianMixture', fontdict=dict(size=20, color='r'))

# 评估模型的似然估计 AIC/BIC  信息准则
models = [GaussianMixture(n_components=n, covariance_type='full', random_state=0).fit(x1) for n in range(1, 21)]
plt.figure(figsize=(12, 8))
bic = np.array([model.bic(x1) for model in models])
aic = np.array([model.aic(x1) for model in models])
plt.plot(np.arange(1, 21), bic, label='BIC')  # 衡量统计模型拟合优良性的一种标准， 取min  模型选择方法
plt.plot(np.arange(1, 21), aic, label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components')
plt.ylabel('score')
plt.xlim(1, 20)
print('\n最优高斯混合数量出现在最小值的位置')
print('BIC:', bic.argmin()+1, '\tAIC:', aic.argmin()+1)

pca = PCA(0.99, whiten=True)  # 保证99%的累计样本方差
data = pca.fit_transform(digits.data)  # (1797,41)

n_components = np.arange(70, 91, 1)
models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(data) for n in n_components]
bics = np.array([model.bic(data) for model in models])
aics = np.array([model.aic(data) for model in models])
plt.figure(figsize=(12, 8))
plt.plot(n_components, bics, label='BIC')
plt.xlim(n_components[0], n_components[-1])
print('BIC:', bics.argmin())  # 找出高斯混合成分数量的粗略估计 72
print('AIC:', aics.argmin())

gmm3 = GaussianMixture(n_components=72, covariance_type='full', random_state=0).fit(data)
print('\n是否收敛:', gmm3.converged_)

# 创建GM模型生成更多的数字
data_new, target = gmm3.sample(64)  # (64, 41) / (64, )
# Generate random samples from the fitted Gaussian distribution. / return X_matrix, y_target
digits_new = pca.inverse_transform(data_new)

fig, ax = plt.subplots(6, 6, figsize=(12, 8), subplot_kw=dict(xticks=[], yticks=[]))
fig.subplots_adjust(hspace=0.05, wspace=0.05)
for i, axi in enumerate(ax.flat):
    imag = axi.imshow(digits_new[i].reshape(8, 8), cmap='binary')
    imag.set_clim(0, 16)

plt.show()
