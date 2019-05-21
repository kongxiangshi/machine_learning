# -*- coding: utf-8 -*-
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

data = fetch_california_housing()
xtr, xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.2, random_state=0)
offset = int(xtr.shape[0]*0.9)
xtr_new = xtr[:offset]
ytr_new = ytr[:offset]
x_val = xtr[offset:]
y_val = ytr[offset:]

train = lgb.Dataset(xtr_new, ytr_new)  # 转化为二进制文件
val = lgb.Dataset(x_val, y_val, reference=train)

params = { 'task': 'train', 'objective': 'regression', 'boosting': 'gbdt',  # 设置提升类型
    'learning_rate': 0.05, 'max_depth': -1, 'metric': {'l2', 'auc'},  # 评估函数
    'num_leaves': 31, 'n_jobs':4, 'feature_fraction': 0.9, 'bagging_fraction': 0.8 ,
    'bagging_freq': 5, # 每5次迭代执行bagging
    'verbose': 1,  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    }

gbm = lgb.train(params=params, train_set=train, num_boost_round=1200, valid_sets=val)
print('Save the model')
gbm.save_model('lightgbn_model.txt')
ypred = gbm.predict(xte)   # 不能为二进制格式   / 使用了早停机制 num_iteration=gbm.best_iteration
mse = mean_squared_error(yte, ypred)
print('MSE:', mse)