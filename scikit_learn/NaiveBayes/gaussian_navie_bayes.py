# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as t_t_s
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture as GM
from sklearn.metrics import accuracy_score as a_s

df = sns.load_dataset('iris')
x = df.drop('species', axis=1)
y = df['species']

xtr, xte, ytr, yte = t_t_s(x, y, test_size=0.25, random_state=0)
print(xtr.shape, yte.shape)  # (112, 4) (38,)
model = GNB()
model.fit(xtr, ytr)
ypred = model.predict(xte)
print("分类准确率：{0:.2%}".format(a_s(yte, ypred)))

# dimensionality reduction
pca = PCA(n_components=2)
new_x = pca.fit_transform(x)
xtr_new, xte_new, ytr_new, yte_new = t_t_s(new_x, y, test_size=0.25, random_state=0)
print(xtr_new.shape, yte_new.shape)  # (112, 2) (38,)
model1 = GNB()
model1.fit(xtr_new, ytr_new)
ypred1 = model1.predict(xte_new)
print("PCA后分类准确率：{0:.2%}".format(a_s(yte_new, ypred1)))
df['PCA1'] = new_x[:, 0]
df['PCA2'] = new_x[:, 1]

sns.set(style='darkgrid')
sns.lmplot('PCA1', 'PCA2', data=df, hue='species', fit_reg=False)

# clustering
model2 = GM(n_components=3, covariance_type='full')
model2.fit(x)
ypred2 = model2.predict(x)
df['cluster'] = ypred2
sns.lmplot('PCA1', 'PCA2', data=df, hue='species', col='cluster', fit_reg=False)

plt.show()