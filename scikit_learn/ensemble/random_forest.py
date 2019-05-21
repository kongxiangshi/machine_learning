# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import BaggingClassifier as BC, RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score


digits = load_digits()

fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(36):
    ax = fig.add_subplot(6, 6, i+1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
    ax.text(0, 0.05, str(digits.target[i]), transform=ax.transAxes)

xtr, xte, ytr, yte = train_test_split(digits.data, digits.target, test_size=0.2, random_state=0)
names = ['DTC', 'BC', 'RFC']
models = [DTC(), BC(n_estimators=300, max_samples=0.8, random_state=0),
          RFC(n_estimators=300, random_state=0)]
for name, model in zip(names, models):
    print('模型{0}的预测准确率：'.format(name), end='\t')
    model.fit(xtr, ytr)
    ypred = model.predict(xte)
    print(accuracy_score(yte, ypred))

plt.show()