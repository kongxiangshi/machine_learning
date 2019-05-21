# -*- coding: utf-8 -*-
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import validation_curve as v_c, learning_curve as l_c
from sklearn.linear_model import LinearRegression


def make_data(n, err=1.0):
    rng = npr.RandomState(1)
    x = rng.rand(n, 1) ** 2
    y = 10 - 1./(x.ravel() + 0.1)
    if err > 0:
        y += err*rng.randn(n)
    return x, y


x, y = make_data(1000)
model = make_pipeline(PolynomialFeatures(), LinearRegression())  # step 预处理, 模型类
degree = np.arange(1, 16)
tr_score, val_score = v_c(model, x, y, param_name='polynomialfeatures__degree',
                      param_range=degree, cv=5, n_jobs=4)
tr_median = np.median(tr_score, axis=1)
val_median = np.median(val_score, axis=1)
print('Optimal degree value:\t', degree[val_median.argmax()])

plt.figure()
plt.plot(degree, tr_median, 'b-', label='Training_score')
plt.plot(degree, val_median, 'r-', label='Validation_score')
plt.legend(loc='lower center', fontsize=14, labelspacing=1, frameon=True)
plt.axis([0, 15, 0, 1.0])
plt.xlabel('Degree')
plt.ylabel('Scores')
plt.title('Validation curve', fontdict=dict(size=20, color='r'))


N , tr_lc, val_lc = l_c(model, x, y, cv=5, n_jobs=4, train_sizes=np.linspace(0.05, 1.0, 20),
                        shuffle=True, random_state=0)
tr_mean = np.mean(tr_lc, 1)
val_mean = np.mean(val_lc, 1)
convergence = np.mean([tr_lc[-1], val_lc[-1]])  # 收敛
print('Convergence score value:\t', convergence)
print(N[0], N[-1])

plt.figure()
plt.plot(N, tr_mean, 'b-', label='Training_score')
plt.plot(N, val_mean, 'r-', label='Validation_score')
plt.hlines(convergence, N[0], N[-1], colors='k', linestyles='--')
plt.axis([N[0], N[-1], 0, 1.0])
plt.xlabel('Training Size')
plt.ylabel('Scores')
plt.title('Learning Curve', fontdict=dict(size=20, color='r'))
plt.text(1000, 0.85, '收敛位置', size=18, color='r', fontproperties='SimHei')
plt.legend(loc='lower center')

plt.show()
