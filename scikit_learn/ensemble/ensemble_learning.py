# -*- coding: utf-8 -*-
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score


data = load_digits()  # dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])
xtr, xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.2)

names = ['ABC', 'GBC', 'BC', 'RFC', 'ETC']
# SAMME.R  base_estimator must support calculation of class probabilities
# estimator_weights_  / estimator_errors_    for each estimator in the boosted ensemble
#  ‘deviance’ (= logistic regression) for classification with probabilistic outputs.
#   For loss ‘exponential’ gradient boosting recovers the AdaBoost algorithm.
models = [AdaBoostClassifier(base_estimator=None, n_estimators=400, learning_rate=0.05, algorithm='SAMME.R'),
          GradientBoostingClassifier(learning_rate=0.01, loss='deviance', n_estimators=500, subsample=0.8,
                                     criterion='friedman_mse', max_features=None, max_depth=5, verbose=0),
          # None = n_features
          BaggingClassifier(base_estimator=None, n_estimators=500, max_samples=1.0, max_features=1.0,
                          bootstrap=True, bootstrap_features=False, oob_score=True, n_jobs=4, verbose=0),
          RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=None, max_features="auto",
                                 bootstrap=True, oob_score=True, n_jobs=4, verbose=0),
          # If “auto”/'sqrt' ,  If “log2”, then max_features=log2(n_features)
          ExtraTreesClassifier(n_estimators=500, criterion='gini', max_depth=None, max_features='sqrt',
                    bootstrap=True, oob_score=True, n_jobs=4, verbose=0, warm_start=False, class_weight=None)]
          # Extremely Randomized Trees
          #  fits a number of randomized decision trees (a.k.a. extra-trees) on various sub-samples of the dataset
          #  Extra-Trees work by decreasing variance while increasing bias.
            # When the randomization is increased above the optimal level,
           # variance decreases slightly while bias increases often significantly.
for name, model in zip(names, models):
    print('model name:', name, end='\t\t')
    model.fit(xtr,ytr)
    if name == 'ABC':
        print('weights:', model.estimator_weights_[-5:], end='\t')
        print('errors:', model.estimator_errors_[-5:])
    if name == 'GBC':
        print('\nTrain score:', model.train_score_[-5:])
        # If subsample == 1 this is the deviance on the training data.
    if name == 'RFC':
        print('\n Feature importance:', model.feature_importances_)
        # the higher, the more important the feature
    ypred = model.predict(xte)
    res = accuracy_score(yte, ypred)
    print('Accuracy score: {0:.3%}'.format(res))