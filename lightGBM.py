# -*- coding: utf-8 -*-
import json
import numpy as np
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_covtype   #   Classes	581012x54   classes 7
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from lightgbm.sklearn import LGBMClassifier


data = fetch_covtype()
print(data.keys())
print(np.unique(data.target))  # [1 2 3 4 5 6 7]
xtr, xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.05, random_state=0)
offset = int(0.9*len(ytr))
xtr_tr, ytr_tr = xtr[:offset], ytr[:offset]
xtr_val, ytr_val = xtr[offset:], ytr[offset:]

'''
lgb_train = lgb.Dataset(data=xtr_tr, label=ytr_tr) # 将数据保存到LightGBM二进制文件将使加载更快   训练集
lgb_validation = lgb.Dataset(xtr_val, ytr_val, reference=lgb_train)  # 验证集
# reference  If this is Dataset for validation, training data should be used as reference

params = {'boosting': 'gbdt',  'num_iterations':1000, 'verbosity':1,
    'objective': 'multiclass', 'metric': 'multi_logloss',  # 评估函数
    'num_leaves': 31,  'learning_rate': 0.1, 'num_threads': 4,'num_class':8,
    'feature_fraction': 0.8, # 建树的特征选择比例
    'bagging_fraction': 0.8, # 建树的样本采样比例
    'bagging_freq': 5,  # k 意味着每k次迭代执行bagging
    }

gbm = lgb.train(params, train_set=lgb_train, num_boost_round=1000, valid_sets=lgb_validation)
y_pred = gbm.predict(xte, num_iteration=gbm.best_iteration) # 如果启用了早期停止，best_iteration方式从最佳迭代中获得预测
# Cannot use Dataset instance for prediction, please use raw data instead
print(y_pred.shape)  # (58102, 8)   返回概率
res = y_pred.argmax(axis=1)
print("Accuracy score (LightGBM): {0:.3%}".format(accuracy_score(yte, res)))  #  93.563%
'''
lgbm = LGBMClassifier(boosting_type='gbdt', num_leaves=33, max_depth=-1, learning_rate=0.1,n_jobs=4, silent=True,
                      n_estimators=500, subsample_for_bin=200000, objective=None, class_weight=None,
                      min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20,
                      subsample=0.8, colsample_bytree=0.8, reg_alpha=0.6, reg_lambda=1.0,
                      random_state=None, importance_type='split')

#  ‘gbdt’, Gradient Boosting Decision Tree. ‘dart’, Dropouts meet Multiple Additive Regression Trees.
#  ‘goss’, Gradient-based One-Side Sampling. ‘rf’, Random Forest.

#   num_leaves (int, optional (default=31)) – Maximum tree leaves for base learners
#   ‘regression’ for LGBMRegressor, ‘binary’ or ‘multiclass’ for LGBMClassifier, ‘lambdarank’ for LGBMRanker.

# you didn't set num_leaves and 2^max_depth > num_leaves
tune_params = {'max_depth':range(2, 6)}
grid = GridSearchCV(lgbm, param_grid=tune_params, n_jobs=4, cv=3, verbose=1).fit(xtr, ytr)
print(grid.best_score_, grid.best_params_)

# scores = cross_val_score(lgbm, xtr, ytr, cv=5, n_jobs=4, verbose=1)
# print('LightGBM(CV=5) Average score', np.mean(scores))