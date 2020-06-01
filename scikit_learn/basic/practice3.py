# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 21:42:28 2018

@author: kxshi
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# text features transform number
sample = ['problem of evil', 'evil queen', 'horizon problem', 'evil love']
vec = CountVectorizer()
x = vec.fit_transform(sample) # sparse matrix 稀疏矩阵 / 特征矩阵转换
df = pd.DataFrame(x.toarray(), columns=vec.get_feature_names())
print(df)
print(df.columns, df.index, df.dtypes)

# term frequency-inverse document frequency 词频逆文档频率
# 单词在文档中出现的频率来衡量其权重, IDF与一个词的常见程度成反比
vec1 = TfidfVectorizer()
x1 = vec1.fit_transform(sample)
df1 = pd.DataFrame(x1.toarray(), columns=vec1.get_feature_names())
print(df1)




