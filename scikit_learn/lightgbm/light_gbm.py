# -*- coding: utf-8 -*-
from lightgbm.sklearn import LGBMRegressor, LGBMClassifier
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV


data = fetch_california_housing()
xtr, xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.2, random_state=0)
offset = int(xtr.shape[0]*0.9)
xtr_new = xtr[:offset]
ytr_new = ytr[:offset]
x_val = xtr[offset:]
y_val = ytr[offset:]

model = LGBMRegressor(boosting_type='gbdt', num_leaves=70, max_depth=7, learning_rate=0.05, n_estimators=2000,
                      subsample_for_bin=200000, subsample=0.7, subsample_freq=0, colsample_bytree=0.8,
                      reg_alpha=0.92, reg_lambda=0.64,
                      random_state=None, n_jobs=4, silent=True, importance_type='split',)

tune_params = {'max_depth':range(64, 100, 5)}
grid = GridSearchCV(estimator=model, param_grid=tune_params, n_jobs=4, iid='ward', refit=True,
                    cv=5, verbose=1).fit(xtr, ytr)
print(grid.best_params_)
print(grid.best_score_)
