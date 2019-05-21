# -*- coding: utf-8 -*-
import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV


data = fetch_california_housing()
xtr, xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.2, shuffle=True, random_state=0)

model00 = CatBoostRegressor(learning_rate=1e-3, max_depth=3, n_estimators=1000,
                            loss_function='RMSE', reg_lambda=0, random_state=0, verbose=0)
model0 = LGBMRegressor(boosting_type='gbdt', num_leaves=31, max_depth=3, learning_rate=1e-3, n_estimators=1000,
                       objective='regression', subsample=0.8, colsample_bytree=0.7, reg_alpha=0,
                       reg_lambda=1., random_state=0, n_jobs=4)
model1 = XGBRegressor(booster='gbtree', max_depth=3, n_estimators=1000, learning_rate=1e-3, silent=True,
                      min_child_weight=1,  objective='reg:linear',  n_jobs=4, gamma=0.1,
                     subsample=0.8, colsample_bytree=0.7, reg_alpha=0, reg_lambda=1, random_state=0)
model2 = GradientBoostingRegressor(learning_rate=1e-3, n_estimators=1000, max_depth=3,
                                   subsample=0.8, loss='ls', verbose=0)
model3 = RandomForestRegressor(n_estimators=1000, criterion='mse',max_depth=3, n_jobs=4, random_state=0,
                               verbose=0)
model4 = LinearRegression(fit_intercept=True, n_jobs=4)
model5 = LassoCV(eps=0.001, n_alphas=200, cv=5,
                 fit_intercept=True, max_iter=1000, tol=1e-4, random_state=0) # 200个alpha中选择一个最优的
model6 = RidgeCV(alphas=tuple(np.linspace(0, 1, 200)),fit_intercept=True, cv=5)

models = [model00, model0, model1, model2, model3, model4, model5, model6]
names = ['CatBoost', 'LightGBM', 'XGBOOST', 'GBDT', 'RF', 'LR', 'Lasso', 'Ridge',]

for name, model in zip(names, models):
    scores = cross_val_score(model, xtr, ytr, cv=5, n_jobs=4, verbose=1)
    print(name, np.mean(scores))