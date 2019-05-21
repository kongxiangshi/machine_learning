# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV as GSCV, RandomizedSearchCV as RSCV
from sklearn.preprocessing import PolynomialFeatures as PF
from sklearn.linear_model import LinearRegression as LR
from sklearn.pipeline import make_pipeline

def make_data(n, err=1.0):
    rng = npr.RandomState(1)
    x = rng.rand(n, 1) ** 2
    y = 10 - 1./(x.ravel() + 0.1)
    if err > 0:
        y += err*rng.randn(n)
    return x, y

x, y = make_data(200)
xtest = np.linspace(0, 1, 500)[:, np.newaxis]

model = make_pipeline(PF(), LR())
tune_params = {'polynomialfeatures__degree':np.arange(20),
               'linearregression__fit_intercept': [True, False],}
grid = GSCV(model, param_grid=tune_params, n_jobs=4, cv=5, verbose=1, refit=True)
grid.fit(x, y)
print(grid.best_params_, grid.best_score_, sep='\t\t')
# {'linearregression__fit_intercept': True, 'polynomialfeatures__degree': 9}
optimal = grid.best_estimator_
ypred = optimal.predict(xtest)

# not all parameter values are tried out, but rather a fixed number of parameter setting
randomized = RSCV(model, param_distributions=tune_params, n_jobs=4, cv=5, verbose=1)
randomized.fit(x, y)
print(randomized.best_estimator_, randomized.best_score_, sep='\t\t')
# steps=[('polynomialfeatures', PolynomialFeatures(degree=6, include_bias=True, interaction_only=False)),
# ('linearregression', LinearRegression(copy_X=True, fit_intercept=False, n_jobs=None, normalize=False))])
optimal1 = randomized.best_estimator_
ypred1 = optimal1.predict(xtest)

plt.style.use('seaborn')
plt.scatter(x.ravel(), y)
plt.plot(xtest, ypred, 'r-', label='GSCV')
plt.plot(xtest, ypred1, 'g-', label='RSCV')
plt.legend(loc='best')
plt.title('Optimalization', fontdict=dict(size=20, color='r'))

plt.show()