# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


data = load_wine()
print(data.keys())  # dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names'])

xtr, xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.2, random_state=1)

model1 = GaussianNB()
model2 = MultinomialNB()  # 主要用于文本分类  基于单词
model3 = ComplementNB()
model4 = BernoulliNB()  # 基于文档

models = [model2, model3, model4]
names = ['MNB', 'CNB', 'BNB']

for name, model in zip(names, models):
    print(name, end='\t:\t')
    scores = cross_val_score(model, xtr, ytr, cv=5, n_jobs=4, verbose=1)
    print(np.mean(scores))

model1.fit(xtr, ytr)
ypred = model1.predict(xte)
score = accuracy_score(yte, ypred)
print('\nAccuracy: {0:.2%}'.format(score))

print(classification_report(yte, ypred))

mat = confusion_matrix(yte, ypred)
plt.figure(figsize=(12, 8))
sns.heatmap(mat, annot=True, fmt='d', annot_kws=dict(size=16), cbar=False, square=True,
            xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel('Predict', fontdict=dict(size=18))
plt.ylabel('True', fontdict=dict(size=18))
plt.title('Confusion Matrix', fontdict=dict(size=24, color='r'))

plt.show()