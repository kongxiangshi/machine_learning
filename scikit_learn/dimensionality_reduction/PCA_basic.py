# -*- coding: utf-8 -*-
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.decomposition import PCA, KernelPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



X = np.array([[-1, -1, 2, 3], [-2, -1, 4, 5], [-3, -2, 4, 3],
              [1, 1, 2, 4], [2, 1, 5, 2], [3, 2, 4, 2]])
y1 = np.array([1, 1, 2, 2, 3, 3])
clf = LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None, n_components=2,
                                 store_covariance=False, tol=0.0001)
# Number of components (< n_classes - 1) for dimensionality reduction
clf.fit(X, y1)
new_x = clf.transform(X)
print(clf.coef_.shape, new_x.shape)  # (3, 4) (6, 2)


sns.set()

digit = load_digits()
df = pd.DataFrame(digit.data)
df['class'] = digit.target
df = df[df['class']<5]
y = df['class']
x = df.drop(['class'], axis=1)

pca = PCA(n_components=2, svd_solver='full')
pca1 = KernelPCA(n_components=2, kernel='rbf')
x_new = pca.fit_transform(x)
x_new1 = pca1.fit_transform(x)
print(x_new.shape)
print(pca.explained_variance_ratio_.sum(), pca.singular_values_)  # 0.9991427833305844  (49,)
# Percentage of variance explained by each of the selected components
print(pca1.lambdas_, pca1.alphas_.shape)  # [2.33113942 1.96473266] / (901, 2)
# Eigenvalues of the centered kernel matrix in decreasing order  lambdas_
# Eigenvectors of the centered kernel matrix. alphas_

xtr, xte, ytr, yte = train_test_split(x_new, y, test_size=0.2, random_state=0)  # (1617, 49)
model = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                             metric='minkowski', n_jobs=None)
model.fit(xtr, ytr)
ypred = model.predict(xte)
plt.figure(figsize=(12,8))
plt.scatter(xte[:,0], xte[:,1], c=ypred, cmap='jet')
plt.colorbar(extend='both', ticks=range(5))
plt.clim(-0.5, 4.5)
plt.show()


xtr1, xte1, ytr1, yte1 = train_test_split(digit.data, digit.target, test_size=0.1, random_state=0)
scores = cross_val_score(model, xtr1, ytr1, cv=5, n_jobs=4, verbose=1)  # 0.9876488772795321
print(scores.mean())  # 0.9876488772795321

# tune_params = {'n_neighbors': range(3, 7), 'weights':['uniform', 'distance']}
# grid = GridSearchCV(model, param_grid=tune_params, n_jobs=4, cv=5, verbose=1).fit(xtr, ytr)
# optimal = grid.best_estimator_
# {'n_neighbors': 5, 'weights': 'uniform'}  0.9876314162028448
