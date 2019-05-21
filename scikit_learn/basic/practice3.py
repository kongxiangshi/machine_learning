# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 21:42:28 2018

@author: kxshi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression as LR
from sklearn.preprocessing import PolynomialFeatures as PF, Imputer
from sklearn.feature_extraction.text import CountVectorizer as CV, TfidfVectorizer as TV

'''
x = np.arange(1, 11)
y = np.array([4, 2, 1, 3, 7, 8, 2, 5, 10, 9])

fig, ax = plt.subplots(1, 2,figsize=(12,8),sharey=True)
ax[0].scatter(x, y)
x_sample = x[:, None]
model = LR().fit(x_sample, y)
yfit = model.predict(x_sample)
ax[0].plot(x, yfit,'-r') # 线性回归曲线
ax[0].set_title('degree=1')

# 更复杂的模型 增加多项式特征
poly = PF(degree=5, include_bias=False) # 5次多项式
x1 = poly.fit_transform(x_sample)  # 衍生特征矩阵
model1 = LR().fit(x1, y)
yfit1 = model1.predict(x1)
ax[1].scatter(x, y)
ax[1].plot(x, yfit1, '-r')
'''
# text features transform number
sample = ['problem of evil', 'evil queen', 'horizon problem', 'evil love']
vec = CV()
x = vec.fit_transform(sample) # sparse matrix 稀疏矩阵 / 特征矩阵转换
df = pd.DataFrame(x.toarray(), columns=vec.get_feature_names())
print(df)
print(df.columns, df.index, df.dtypes)

# term frequency-inverse document frequency 词频逆文档频率
# 单词在文档中出现的频率来衡量其权重, IDF与一个词的常见程度成反比
vec1 = TV()
x1 = vec1.fit_transform(sample)
df1 = pd.DataFrame(x1.toarray(), columns=vec1.get_feature_names())
print(df1)




