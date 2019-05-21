# -*- coding: utf-8 -*-

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, RidgeCV, ElasticNet, ElasticNetCV
from sklearn.metrics import mean_squared_error


data = load_boston()  # dict_keys(['data', 'target', 'feature_names', 'DESCR', 'filename'])  (506, 13) (506,)
xtr, xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.2, random_state=0)
models = [LinearRegression(fit_intercept=True, normalize=False,  n_jobs=4),
          Lasso(alpha=1.0, fit_intercept=True, normalize=False, selection='cyclic'),
          Ridge(alpha=1.0, fit_intercept=True, normalize=False, solver='auto'),
          ElasticNet(alpha=1.0, l1_ratio=0.5), ]
names = ['LR', 'Lasso', 'Ridge', 'ElasticNet']
for model, name in zip(models, names):
    print('model name:\t{0}'.format(name), end='\t||\t')
    model.fit(xtr, ytr)
    ypred = model.predict(xte)
    res = mean_squared_error(yte, ypred)
    print('MSE: {0:.2f}'.format(res))

print()
model1 = Lasso()
tune_params = {'alpha':[0.15, 0.18, 0.168, 0.17, 0.158]}
optimal = GridSearchCV(model1, param_grid=tune_params, cv=5, n_jobs=4).fit(xtr, ytr)
print(optimal.best_params_)  # {'alpha': 0.168}
optimal_model = optimal.best_estimator_
ypred1 = optimal_model.predict(xte)
res1 = mean_squared_error(yte, ypred1)
print('Optimal Lasso MSE', res1)

# The best model is selected by cross-validation
model2 = LassoCV(eps=0.0001, n_alphas=500, cv=5, n_jobs=4)  # 类似于网格搜索最优alpha eps=min_alpha/max_alpha
model2.fit(xtr, ytr)
# model2.alphas_ : The grid of alphas used for fitting
print(model2.alpha_)  # The amount of penalization chosen by cross validation  0.16836594224865464
ypred2 = model2.predict(xte)
res2 = mean_squared_error(yte, ypred2)
print('LassoCV MSE', res2)

print()
model3 = Ridge(alpha=1.0, solver='auto')
tune_params3 = {'alpha':[0.001, 0.005, 0.01, 0.015,]}
optimal3 = GridSearchCV(model3, param_grid=tune_params3, cv=5, n_jobs=4).fit(xtr, ytr)
print(optimal3.best_params_)  # {'alpha': 0.001}
optimal_model3 = optimal3.best_estimator_
ypred3 = optimal_model3.predict(xte)
res3 = mean_squared_error(yte, ypred3)
print('Optimal Ridge MSE', res3)

# The best model is selected by cross-validation
model4 = RidgeCV(alphas=(0.0, 1.0, 300.0), cv=5)  # 类似于网格搜索最优alpha
model4.fit(xtr, ytr)
print(model4.alpha_)  # The amount of penalization chosen by cross validation  0.0
ypred4 = model4.predict(xte)
res4 = mean_squared_error(yte, ypred4)
print('RidgeCV MSE', res4)

print()
model5 = ElasticNet()
tune_params5 = {'alpha':[0.11, 0.13, 0.15, 0.17, 0.19], 'l1_ratio':[0.1, 0.3, 0.5, 0.7, 0.9]}
optimal5 = GridSearchCV(model5, param_grid=tune_params5, cv=5, n_jobs=4).fit(xtr, ytr)
print(optimal5.best_params_)  # {'alpha': 0.15, 'l1_ratio': 0.9}
optimal_model5 = optimal5.best_estimator_
ypred5 = optimal_model5.predict(xte)
res5 = mean_squared_error(yte, ypred5)
print('Optimal ElasticNet MSE', res5)

# The best model is selected by cross-validation
model6 = ElasticNetCV(l1_ratio=0.5, eps=0.0001, n_alphas=500, cv=5, n_jobs=4)  # 类似于网格搜索最优alpha eps=min_alpha/max_alpha
model6.fit(xtr, ytr)
print(model6.alpha_)  # The amount of penalization chosen by cross validation  0.160
ypred6 = model6.predict(xte)
res6 = mean_squared_error(yte, ypred6)
print('ElesticCV MSE', res6)
