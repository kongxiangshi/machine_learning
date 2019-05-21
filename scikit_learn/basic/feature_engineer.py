# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures as PF, OneHotEncoder as OHE
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.feature_extraction.text import CountVectorizer as  CV, TfidfVectorizer as TV
from sklearn.linear_model import LinearRegression as LR
from sklearn.impute import SimpleImputer as SI
from sklearn.pipeline import make_pipeline


data= [{'A': 'apple'}, {'A':'apple'}, {'A':'banana'}, {'A':'peach'}]
vec = DV(sparse=False, dtype=np.int_)
res = vec.fit_transform(data)
print(res)

data1 = pd.DataFrame({'A':['apple', 'apple', 'banana', 'peach']})
data2 = pd.DataFrame({'B':['apple', 'peach']})
enc = OHE()
enc.fit(data1)
print(enc.categories_)  # [array(['apple', 'banana', 'peach'], dtype=object)]
print(enc.transform(data2).toarray())  # [[1. 0. 0.] [0. 0. 1.]]

sample = ['problem of evil', 'evil queen', 'horizon problem']
vec1 = CV()
x =vec1.fit_transform(sample)
print(pd.DataFrame(x.toarray(), columns=vec1.get_feature_names()))

vec2 = TV()
x2 = vec2.fit_transform(sample)
print(pd.DataFrame(x2.toarray(), columns=vec2.get_feature_names()))
print(1/3 * np.log(3/2), 1/3*np.log(3/1))  # tf*idf  ln计算  归一化

x2 = np.arange(1, 6)
y2 = np.array([4, 2, 1, 3, 7])
model = LR().fit(x2[:, None], y2)
ypred1 = model.predict(x2[:, np.newaxis])

poly = PF(degree=3, include_bias=False, interaction_only=False)
x2_new = poly.fit_transform(x2[:, None])  # 衍生特征
model.fit(x2_new, y2)
ypred2 = model.predict(x2_new)

poly = PF(degree=2, include_bias=True, interaction_only=False)
x3_new = poly.fit_transform(x2[:, None])  # 衍生特征
model.fit(x3_new, y2)
ypred3 = model.predict(x3_new)

plt.scatter(x2, y2)
plt.plot(x2, ypred1, 'r-', label='Original')
plt.plot(x2, ypred2, 'g-', label='Degree=3')
plt.plot(x2, ypred3, 'b-', label='Degree=2')
plt.legend(loc='best')
plt.title('衍生特征', fontdict=dict(size=20, color='r'), fontproperties='SimHei')


x4 = np.array([[np.nan, 0, 3], [3, 7, 9], [4, np.nan, 6]])
y4 = np.array([14, 16, 8])
imp = SI(strategy='median')
x4_new = imp.fit_transform(x4)
print(x4_new)


model1 = make_pipeline(SI(strategy='mean'), PF(degree=3), LR())
model1.fit(x4_new, y4)
ypred4 = model1.predict(x4_new)
print(ypred4)

plt.show()