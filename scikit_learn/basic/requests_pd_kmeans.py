# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 09:34:43 2018

@author: kxshi
"""
'''
import requests
import re
import pandas as pd

def retrieve_dji_list(url):
    r=requests.get(url)
    search_pattern=re.compile('class="wsod_symbol">(.*?)<\/a>.*?<span.*?">(.*?)<\/span>.*?\n.*?class="wsod_stream">(.*?)<\/span>') #模式匹配
    dji_list_in_text=re.findall(search_pattern,r.text) #所有符合要求的集合到list
    dji_list=[]
    for item in dji_list_in_text:
        dji_list.append([item[0],item[1],float(item[2])]) #部分添加到list，类型转换str——float
    return dji_list
url='http://money.cnn.com/data/dow30/'
dji_list=retrieve_dji_list(url)
djidf=pd.DataFrame(dji_list)#转化为DataFrame格式
cols=['code','name','lasttrade']#replace 列名
djidf.columns=cols
print(djidf)
'''
#基于10只道琼斯指数成分股一年来相邻2天的收盘价close涨跌数据规律对它们进行聚类
#抓取url，解析re，下载为list是s，转化为dataframe格式，字段名分类 ，返回df1
#只选取close字段，判断相邻2天涨跌，返回0，1，-1，
#分为3类，
import requests
import re
import json
import pandas as pd
#from datetime import date
from sklearn.cluster import KMeans
import numpy as np

def geturl(stock_code):
    quotes=[]
    url='http://finance.yahoo.com/quote/%s/history?p=%s'%(stock_code,stock_code)#%s占位符
    r=requests.get(url)
    m=re.findall('"HistoricalPriceStore":{"prices":(.*?),"isPending"',r.text)#re格式
    if m:
        quotes=json.loads(m[0]) #json格式下载——dic key-value
        quotes=quotes[::-1] #倒排分片
    return [item for item in quotes if not 'type' in item]#如果项目中没有“类型”，则返回引文项
def create_df(stock_code):
    quotes=geturl(stock_code)
    list1=['close','date','high','low','open','volume']
    df1=pd.DataFrame(quotes,columns=list1)
    df1=df1.fillna(df1.mean())#JUNZHImean代替NaN
    return df1
listDji=['MMM','AXP','AAPL','BA','CAT','CVX','CSCO','KO','DIS','DD']
listTemp=[0]*len(listDji)
for i in range(len(listTemp)):
    listTemp[i]=create_df(listDji[i]).close #只选取listDji中的close字段
status=[0]*len(listDji)
for i in range(len(status)):
    status[i]=np.sign(np.diff(listTemp[i])) #相邻2天的差价，判断涨跌
kmeans=KMeans(n_clusters=3).fit(status)#分为3类
pred=kmeans.predict(status)#确定结果归类
print(pred)
'''
quotes=geturl('IBM')
list1=[]
for i in range(len(quotes)):
    x=date.fromtimestamp(quote[i]['date'])#时间戳转化为当前时间
    y=date.strftime(x,'%Y-%m-%d')#时间格式转换
    list1.append(y)
quotesdf_ori=pd.DataFrame(quotes,index=list1)
quotesdf=quotesdf_ori.drop(['date'],axis=1)#del data列 
print(quotesdf)
'''