# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
from xgboost.sklearn import XGBClassifier


df = pd.read_csv('pd_speech_features.csv', header=1)  # (756, 755)
print(np.sum(df.isnull().sum() == 0))  # 判断是否有缺失值

target = df['class']
feature_matrix = df.drop(['class','id'], axis=1)

xtr, xte, ytr, yte = train_test_split(feature_matrix, target, test_size=0.15, random_state=0)

'''
models = [RandomForestClassifier(n_estimators=400, max_depth=15, max_features=0.7, bootstrap=True, oob_score=True, n_jobs=4),
          GradientBoostingClassifier(n_estimators=400)]
names = ['GBC', 'RFC']

model2 = GaussianNB()  # high dimension 
score2 = cross_val_score(model2, xtr, ytr, cv=5, n_jobs=4)
print('GaussianNB: {0:.3%}'.format(np.mean(score2)))  #  GNB: 74.302%

print('\nNo PCA:')    # GBC 88.469%  / RFC 88.626%
for model, name in zip(models, names):
    print('\tmodel name: ' + name + '\t||\t', end='')
    scores = cross_val_score(model, xtr, ytr, cv=10, n_jobs=4)   # model.oob_score_
    average = np.mean(scores)
    print('Average_score(cv=10): {0:.3%}'.format(average))

    model.fit(xtr, ytr)
    ypred = model.predict(xte)
    print('Classification report:')
    print(classification_report(yte, ypred))

print('\nPCA:')
matrix_ls = []
explained_variance_ratio = [50, 100, 150]
for value in explained_variance_ratio:
    new_matrix = PCA(n_components=value, svd_solver='full').fit_transform(feature_matrix)
    #  the amount of variance that needs to be explained   If 0 < n_components < 1 and svd_solver == 'full'
    matrix_ls.append(new_matrix)

for matrix, var in zip(matrix_ls, explained_variance_ratio):
    print('\nExplained_variance_ratio = {0}'.format(var))
    xtr1, xte1, ytr1, yte1 = train_test_split(matrix, target, test_size=0.15, random_state=0)
    print(matrix.shape)
    for model, name in zip(models, names):
        print('\tmodel name: ' + name + '\t||\t', end='')
        scores1 = cross_val_score(model, xtr1, ytr1, cv=10, n_jobs=4)
        average1 = np.mean(scores1)
        print('Average_score(cv=10): {0:.3%}'.format(average1))


dtrain = xgb.DMatrix(xtr, label=ytr)
dtest = xgb.DMatrix(xte, label=None)

params = {'max_depth':3, 'eta':0.1, 'objective':'binary:logistic', 'eval_metric':'error'}

cv = xgb.cv(params, dtrain, num_boost_round=70, nfold=3, early_stopping_rounds=20, show_stdv=False,
           shuffle=True, verbose_eval=5)
cv[['train-error-mean', 'test-error-mean']].plot()

bst = xgb.train(params, dtrain, num_boost_round=70)
ypred2 = bst.predict(dtest)  # 概率
df2 = pd.Series(ypred2)
res = df2.apply(lambda x : 1 if x>0.5 else 0)
print('Accuracy score(XGBOOST): {0:.3%}'.format(accuracy_score(yte, res)))

plt.show()
'''

model3 = XGBClassifier(max_depth=9, learning_rate=0.05, n_estimators=300, objective='binary:logistic',
                       booster='gbtree', n_jobs=4, gamma=0.1, subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
                       reg_alpha=0, reg_lambda=1, base_score=0.5, random_state=0, missing=None, importance_type='gain')
model3.fit(xtr, ytr, eval_metric='logloss', verbose=True)
ypred3 = model3.predict(xte)
print('Accuracy score(XGBClassifier): {0:.3%}'.format(accuracy_score(yte, ypred3)))  # 86.842%

tune_params = { 'n_estimators': [200, 300]}
# {'max_depth': 9} 0.8831 取值最好在3-10之间   {'min_child_weight': 1} 0.88317
# {'gamma': 0.1} 0.88317    {'reg_alpha': 0} 0.88317, {'reg_lambda': 0.7} 0.88317
#  {'learning_rate': 0.05} 0.88317  {'n_estimators': 300} 0.8862928348909658
grid = GridSearchCV(model3, param_grid=tune_params, cv=3, n_jobs=4, verbose=2)
grid.fit(xtr, ytr)
print(grid.best_params_, grid.best_score_)
