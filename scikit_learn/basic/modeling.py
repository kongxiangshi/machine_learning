# -*- coding: utf-8 -*-
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier
from lightgbm.sklearn import LGBMClassifier

data = load_breast_cancer()
print(data.keys())  # dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])
xtr, xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.2, random_state=1)
print(xtr.shape, xte.shape, ytr.shape, yte.shape)  # (120, 4) (30, 4) (120,) (30,)

model1 = LogisticRegression(penalty='l2', solver='lbfgs', verbose=0, n_jobs=4, multi_class='auto', max_iter=3000)
model2 = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None)
model3 = SVC(C=1.0, kernel='rbf', gamma='scale')
model4 = GaussianNB()
model5 = KNeighborsClassifier(n_neighbors=8, n_jobs=4)
model6 = RandomForestClassifier(n_estimators=300, max_depth=5, bootstrap=True, oob_score=True, n_jobs=4)
model7 = ExtraTreesClassifier(n_estimators=300, max_depth=5, bootstrap=True, oob_score=True, n_jobs=4)
model8 = GradientBoostingClassifier(loss='deviance', learning_rate=0.01, n_estimators=300, subsample=0.8,
                                    max_depth=5)
model9 = XGBClassifier(max_depth=5, learning_rate=0.001, n_estimators=300, objective='binary:logistic',
                       booster='gbtree', n_jobs=4, subsample=0.8, colsample_bytree=0.8,
                       reg_lambda=1, reg_alpha=0)
model10 = LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.01, n_estimators=300,
                         objective=None, subsample=0.8, colsample_bytree=0.8, reg_alpha=0, reg_lambda=0,
                         n_jobs=4, importance_type='split')


models = [model1, model2, model3, model4, model5, model6, model7, model8, model9, model10]
names = ['LogisticRegression', 'DecisionTree', 'SVM', 'Naive_bayes', 'KNN', 'RandomForest', 'ExtraTrees',
         'GradientBoosting', 'XGBoost', 'LightGBM']
print('\nAccuracy:')
for name, model in zip(names, models):
    model.fit(xtr, ytr)
    ypred = model.predict(xte)
    print('{0} : {1:.2%}'.format(name, accuracy_score(yte, ypred)))

