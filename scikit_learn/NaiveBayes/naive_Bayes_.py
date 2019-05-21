# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 19:36:08 2018

@author: kxshi
"""

import numpy.random as npr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs, fetch_20newsgroups
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def gaussian():
    x, y = make_blobs(n_samples=500, n_features=2, centers=2, cluster_std=1.2, random_state=1)
    # Label的数据服从高斯分布(正态分布)-钟形曲线
    rng = npr.RandomState(0)
    xtest = [-6, -14] + [14, 18] * rng.rand(200, 2)

    model = GaussianNB(priors=None, var_smoothing=1e-9)
    model.fit(x, y)
    print('mean of each feature:', model.theta_)
    print('variance of each feature:', model.sigma_)
    print('absolute additive value to variance:', model.epsilon_)
    ypred = model.predict(xtest)
    yprob = model.predict_proba(xtest)  # 计算样本属于某个标签的概率(后验概率)
    size = 80* yprob**2
    plt.scatter(xtest[:, 0], xtest[:, 1], c=ypred, cmap='RdBu', s=size, alpha=0.7)
    plt.axis([-6, 8, -14, 4])

gaussian()

data = fetch_20newsgroups()  # dict_keys(['data', 'filenames', 'target_names', 'target', 'DESCR'])
print(len(data.target_names))  # 20

content = data.target_names[0: 21: 2]
train = fetch_20newsgroups(subset='train', categories=content)  # 选取多类新闻 作为训练集/测试集
test = fetch_20newsgroups(subset='test', categories=content)

print(len(train.data), len(test.data), type(train.data[0]))  # 2153 / 1432  list  str
print(pd.Series(train.target).unique())  # [3 2 1 5 9 4 8 0 7 6]

model1 = make_pipeline(TfidfVectorizer(), MultinomialNB())
model2 = make_pipeline(TfidfVectorizer(), ComplementNB())
model3 = make_pipeline(TfidfVectorizer(), BernoulliNB())
names = ['MNB', 'CNB', 'BNB']
models = [model1, model2, model3]
for name, model in zip(names, models):
    print(name)
    model.fit(train.data, train.target)
    labels = model.predict(test.data)
    acc = accuracy_score(test.target, labels)
    print('模型准确率为{0:.2%}'.format(acc))


train1 = fetch_20newsgroups(subset='train', categories=data.target_names)  # 选取多类新闻 作为训练集/测试集
test1 = fetch_20newsgroups(subset='test', categories=data.target_names)

model_new = make_pipeline(TfidfVectorizer(), ComplementNB()).fit(train1.data, train1.target)  # step
labels_new = model_new.predict(test1.data)
print(classification_report(test1.target, labels_new, target_names=train1.target_names))

mat = confusion_matrix(test1.target, labels_new)
plt.figure(figsize=(12, 8))
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            annot_kws={'size':12, 'weight':'bold'},
            xticklabels=train1.target_names, yticklabels=train1.target_names)
plt.xlabel('Predict', fontdict=dict(size=16, color='r'))
plt.ylabel('True', fontdict=dict(size=16, color='r'))
plt.title('Confusion Matrix', fontdict=dict(size=20, color='r'))


def predict_category(s, train=train1, model=model_new):
    ypred = model.predict([s])
    print('{0}\t所属类别：\t{1}'.format(s, train.target_names[ypred[0]]))

predict_category('screen resolution')  # 预测所属新闻类别
predict_category('sending a payload to the ISS')
predict_category('love and peace')

plt.show()