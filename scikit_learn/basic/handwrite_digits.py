# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score as c_v_s, train_test_split as t_t_s
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.manifold import Isomap

digits = load_digits()
print(digits.keys())  # dict_keys(['data', 'target', 'target_names', 'images', 'DESCR'])
print(digits.data.shape, digits.images.shape, digits.target.shape)  # (1797, 64) (1797, 8, 8) (1797,)

fig, ax = plt.subplots(6, 6, subplot_kw=dict(xticks=[], yticks=[]), gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, axi in enumerate(ax.flat):
    axi.imshow(digits.images[i], cmap='binary')  # 插补 interpolation='nearest'
    axi.text(0.05, 0.05, str(digits.target[i]), color='g', transform=axi.transAxes)

# dimensionality reduction
iso = Isomap(n_components=2)
new = iso.fit_transform(digits.data)
print(new.shape)
sns.set(style='whitegrid')
plt.figure()
plt.scatter(new[:, 0], new[:, 1], c=digits.target, cmap=plt.cm.get_cmap('Spectral', 10),
            edgecolor='none', alpha=0.6)
plt.colorbar(label='Digits', ticks=range(10), extend='both')
plt.clim(-0.5, 9.5)

# classification
model = RFC(n_estimators=400)
xtr, xte, ytr, yte = t_t_s(digits.data, digits.target, test_size=0.2, random_state=0)
model.fit(xtr, ytr)
ypred = model.predict(xte)
fig, ax = plt.subplots(10, 10, figsize=(14, 10), subplot_kw={'xticks':[], 'yticks':[]},
                       gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, axi in enumerate(ax.flat):
    axi.imshow(xte.reshape(-1, 8, 8)[i], cmap='binary')
    axi.text(0.05, 0.05, str(ypred[i]), color='g' if (ypred[i]==yte[i]) else 'r', transform=axi.transAxes)

# model analysis
names = ['GNB', 'DTC', 'SVC']
models = [GNB(), DTC(), SVC(gamma='scale')]
for name, model in zip(names, models):
    scores = c_v_s(model, digits.data, digits.target, cv=5, n_jobs=4, verbose=1)
    print(name + ':\t', scores)
    print('cross validation mean score: {0:.2f}'.format(scores.mean()))

plt.show()