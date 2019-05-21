# -*- coding: utf-8 -*-
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import  seaborn as sns
from sklearn.linear_model import LinearRegression as LR

sns.set(style='whitegrid')
rng = npr.RandomState(42)
x = 10 * rng.rand(50)
y = 2 * x - 1 + rng.randn(50)

model = LR(fit_intercept=True, n_jobs=4, normalize=False)
model.fit(x[:, np.newaxis], y)
print("曲线的斜率：{0}\t\t曲线的截距：{1}".format(model.coef_[0], model.intercept_))
xfit = np.linspace(0, 10, 500)
ypred = model.predict(xfit[:, None])

plt.rc('font', family='SimHei', size=16)
plt.figure(figsize=(12, 8))
plt.scatter(x, y)
plt.plot(xfit, ypred, '-r', lw=2)
plt.text(5, 7.5, r'拟合回归曲线')  # fontproperties='SimHei'
plt.title('LinearRegression', fontdict=dict(size=24, color='r'))
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('tight')
plt.show()