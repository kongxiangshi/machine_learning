# -*- coding: utf-8 -*-

from sklearn.datasets import make_blobs, make_regression
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.svm import LinearSVR, SVR, NuSVR
from sklearn.metrics import accuracy_score, mean_squared_error



x, y = make_blobs(n_samples=1000, n_features=20, centers=4, cluster_std=0.8, shuffle=True, random_state=0)
x1, y1 = make_regression(n_samples=1000, n_features=100, n_informative=10, n_targets=1, random_state=0)

xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2, random_state=1)
xtr1, xte1, ytr1, yte1 = train_test_split(x1, y1, test_size=0.2, random_state=1)

model1 = LinearSVC(penalty='l2', loss='squared_hinge', dual=False, C=1.0, verbose=0)
model2 = SVC(C=1.0, kernel='rbf', gamma='scale', verbose=False)
model3 = NuSVC(nu=0.5, kernel='rbf', gamma='scale')

model4 = LinearSVR(epsilon=0.0, C=1.0, loss='epsilon_insensitive', fit_intercept=True, dual=True, verbose=0)
model5 = SVR(kernel='rbf', gamma='scale', C=1.0, epsilon=0.1, verbose=False)
model6 = NuSVR(nu=0.5, kernel='rbf', gamma='scale', C=1.0)

model_classification= [model1, model2, model3]
model_regression = [model4, model5, model6]
names1 = ['LSVC', 'SVC', 'NuSVC']
names2 = ['LSVR', 'SVR', 'NuSVR']
len = list(range(0, 3))
print('Classification')
for i, name, model in zip(len, names1, model_classification):
    print(name)
    model.fit(xtr, ytr)
    if i != 0:
        print(model.support_vectors_.shape)
    ypred = model.predict(xte)
    res = accuracy_score(yte, ypred)
    print('Accuracy:\t{0:.2%}'.format(res))

print('\nRegression')
for i, name, model in zip(len, names2, model_regression):
    print(name)
    model.fit(xtr1, ytr1)
    if i != 0:
        print(model.support_vectors_.shape)
    ypred1 = model.predict(xte1)
    res = mean_squared_error(yte1, ypred1)
    print('MSE:\t{0:.3f}'.format(res))

