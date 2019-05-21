# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 13:18:20 2018

@author: kxshi
"""
import math
import numpy as np
import pandas as pd


'''
def cal_entropy(n, yes):
    """计算初始熵"""
    p = yes/n
    res = -p*math.log(p,2)-(1-p)*math.log(1-p,2)
    return res

# id3
def id3(c1, y1, c2, y2, c3=0, y3=0):
    """计算条件熵（分为3个子属性,c1/yes），若分2类，c3=0"""
    c = c1 + c2 + c3
    if c1 == y1 or y1 == 0:
        s1 =0
    else:
        s1 = cal_entropy(c1, y1)
    if c2 == y2 or y2 == 0:
        s2 = 0
    else:
        s2 = cal_entropy(c2, y2)
    if c3 != 0:
        s3 = cal_entropy(c3, y3)
        s4 = c1/c*s1 + c2/c *s2 + c3/c*s3
    else:
        s4 = c1/c*s1 + c2/c *s2
    return s4


# C4.5
def gain_ratio(gain, c1, c2, c3=0):
    """信息增益率  gain/split_info  c1为属性A的a1数目"""
    # a1=17,a2=5,a3=0  与判定类别（yes/no）无关
    c = c1 + c2 +c3  
    a1, a2, a3 = c1/c, c2/c, c3/c
    if c3 != 0:
        res = -a1*math.log(a1, 2) - a2*math.log(a2, 2) - a3*math.log(a3, 2)
    else:
        res = -a1*math.log(a1, 2) - a2*math.log(a2, 2)
    result = gain/res
    return res, result

    
samples = 14
num_yes = 9
r = cal_entropy(samples, num_yes)
print('初始熵:{0:.3f}\n'.format(r))
ls = [(5, 2, 4, 4, 5, 3), (4, 2, 6, 4, 4, 3), (6, 3, 8, 6, 0, 0)]
# a1 yes / a2 yes / a3 yes
for i, item in enumerate(ls):
    print('第{0}个属性:'.format(i+1))
    r1 = id3(item[0], item[1], item[2], item[3], item[4], item[5])
    print('条件熵(属性中3个子属性):{0:.3f}'.format(r1))
    gain = r-r1
    print('信息增益：{0:.3f}'.format(gain))  # 取最大gain作为判别属性
    n, res = gain_ratio(gain, item[0], item[2], item[4])
    print(n)
    print('信息增益率：{0:.3f}\n'.format(res)) # # 取最大gain_ratio作为判别属性
'''

def naive_bayes(n, yes, c1, c1_yes, c2, c2_yes, c3, c3_yes,):
    p1 = yes/n
    p1c1 = c1_yes/yes
    p1c2 = c2_yes/yes
    p1c3 = c3_yes/yes
    p_yes = p1*p1c1*p1c2*p1c3
    
    no = n-yes
    p2 = no/n
    p2c1 = (c1-c1_yes)/no
    p2c2 = (c2-c2_yes)/no
    p2c3 = (c3-c3_yes)/no
    p_no = p2*p2c1*p2c2*p2c3
    
    return p_yes, p_no

pyes, pno = naive_bayes(10, 7, 7, 4, 4, 2, 5, 4,)
# samples=14, yes_num=9
# c1_sunny=5, c1_sunny_yes=2 / c2_cool=4,c2_cool_yes=3/... c3, c4
print('pyes:{0:.4f}\tpno:{1:.4f}'.format(pyes, pno)) 
# 选较大值argmax = p(C)*p(X|C) 归类   P(X|C) —— C发生的条件下, 样本X发生的概率
# 先验概率 / 后验概率（条件概率）

'''
# K-mean聚类
x = np.array([[2, 68], [27, 83], [30, 61], 
                 [62, 69],[29, 68],[62,90],
                 [75,62],[24,43]])
x1 = np.array([[2, 68]]) #初始簇中心点（2个）
x2 = np.array([[27, 83]])

def dist(x_move, x):
    dist1 = ((x_move-x)**2).sum(axis=1) 
    res = np.sqrt(dist1)  
    return res

res1 = dist(x1, x)
res2 = dist(x2, x)
index = [chr(i) for i in range(65, 65+8)]
df = pd.DataFrame(dict(M1 = res1, M2=res2), index=index )
df1 = df.mean()
df['comparison'] = (df['M1']<df['M2']) * 1 # boolean 
df['comparison'] = df['comparison'].map({1:'M1', 0:'M2'}) # 映射mapping
print(df.sort_values(by='M1'))

df.index = [ord(char)-65 for char in  df.index]
print(df.sort_values(by='M1'))

x_new = x[[2,4,7,0,1,8]]
print(x_new.mean(axis=0))
x_new1 = x[[3,10,9,5,6,11]]
print(x_new1.mean(axis=0))
'''