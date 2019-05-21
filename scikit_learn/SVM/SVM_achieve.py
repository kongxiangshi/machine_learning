# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 13:14:59 2018

@author: kxshi
"""
import numpy as np
import random

def load_dataset(filename): 
    ls = []
    label, data = [], []
    with open(filename, 'r', encoding='utf-8') as f:
        for item in f.readlines():
            if item != '\n':
                ls.append(item)
        for item in ls:
            new = item.strip('\n').split(',')
            label.append(eval(new[2]))
            data.append([eval(new[0]), eval(new[1])])
    return data, label  #返回数据特征和类别

filename = 'c:/users/kxshi/desktop/sampledata.txt'
data, label = load_dataset(filename)


def select_rand(i, m): #在0-m中随机选择一个不是i的整数
    j = i
    while  j == i:
        j = random.randint(0, m)
    return j


def clip_alpha(a, H, L):  # 保证a在L和H范围内（L <= a <= H）
    if a > H:
        a = H
    if a < L:
        a = L
    return a


def kernel(X, A, kTup): 
    # 核函数, 输入参数X:支持向量的特征树；A：某一行特征数据；
    # kTup：('lin',k1)核函数的类型和参数
    m, n = np.shape(X) 
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin': #线性函数
        K = X * A.T  # 转置
    elif kTup[0] == 'rbf': # 径向基函数(radial bias function)
        for j in range(m):  # row
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = np.exp(K/(-1*kTup[1]**2)) 
    else:
        raise NameError('Houston We Have a Problem -That Kernel is not recognized')
    return K


class OptStruct:
    # 定义存储
    def __init__(self, data_fea, labels, C, toler, kTup):  # 存储各类参数
        self.X = data_fea   # 数据特征
        
        self.labelMat = labels # 数据类别
        self.C = C  # 软间隔参数C，参数越大，非线性拟合能力越强
        self.tol = toler # 停止阀值
        self.m = np.shape(data_fea)[0] # 数据行数
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0 #初始设为0
        self.eCache = np.mat(np.zeros((self.m, 2))) #缓存
        self.K = np.mat(np.zeros((self.m, self.m))) #核函数的计算结果
        for i in range(self.m):
            self.K[:,i] = kernel(self.X, self.X[i,:], kTup)  # 调用函数
            

def cal_Ek(OS, k): #计算Ek（参考《统计学习方法》p127公式7.105）
    fXk = float(np.multiply(OS.alphas, OS.labelMat).T * OS.K[:, k] + OS.b) # 类实例化
    Ek = fXk - float(OS.labelMat[k])
    return Ek

#随机选取aj，并返回其E值
def select_j(i, OS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    OS.eCache[i] = [1, Ei]
    validEcacheList = np.nonzero(OS.eCache[:, 0].A)[0]  #返回矩阵中的非零位置的行数
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = cal_Ek(OS, k)  # 函数调用
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE: # 返回步长最大的aj
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = select_rand(i, OS.m) #调用函数
        Ej = cal_Ek(OS, j)
    return j, Ej


def update_Ek(OS, k): #更新os数据
    Ek = cal_Ek(OS, k)
    OS.eCache[k] = [1, Ek]


#首先检验ai是否满足KKT条件，如果不满足，随机选择aj进行优化，更新ai,aj,b值
def innerL(i, OS): #输入参数i和所有参数数据
    Ei = cal_Ek(OS, i) #计算E值
    if ((OS.labelMat[i]*Ei < -OS.tol) and (OS.alphas[i] < OS.C)) or ((OS.labelMat[i]*Ei > OS.tol) and (OS.alphas[i] > 0)): 
        # 检验这行数据是否符合KKT条件 参考《统计学习方法》p128公式7.111-113
        j, Ej = select_j(i, OS, Ei) #随机选取aj，并返回其E值
        alphaIold = OS.alphas[i].copy()
        alphaJold = OS.alphas[j].copy()
        if (OS.labelMat[i] != OS.labelMat[j]): #以下代码的公式参考《统计学习方法》p126
            L = max(0, OS.alphas[j] - OS.alphas[i])
            H = min(OS.C, OS.C + OS.alphas[j] - OS.alphas[i])
        else:
            L = max(0, OS.alphas[j] + OS.alphas[i] - OS.C)
            H = min(OS.C, OS.alphas[j] + OS.alphas[i])
        if L == H:
            print("L==H")
            return 0
        eta = 2.0 * OS.K[i,j] - OS.K[i,i] - OS.K[j,j] #参考《统计学习方法》p127公式7.107
        if eta >= 0:
            print("eta>=0")
            return 0
        OS.alphas[j] -= OS.labelMat[j]*(Ei - Ej)/eta #参考《统计学习方法》p127公式7.106
        OS.alphas[j] = clip_alpha(OS.alphas[j],H,L) #参考《统计学习方法》p127公式7.108
        update_Ek(OS, j)
        if (abs(OS.alphas[j] - alphaJold) < OS.tol): #alpha变化大小阀值（自己设定）
            print("j not moving enough")
            return 0
        OS.alphas[i] += OS.labelMat[j]*OS.labelMat[i]*(alphaJold - OS.alphas[j])
        #参考《统计学习方法》p127公式7.109
        update_Ek(OS, i) #更新数据
        #以下求解b的过程，参考《统计学习方法》p129公式7.114-7.116
        b1 = OS.b - Ei- OS.labelMat[i]*(OS.alphas[i]-alphaIold)*OS.K[i,i] - OS.labelMat[j]*(OS.alphas[j]-alphaJold)*OS.K[i,j]
        b2 = OS.b - Ej- OS.labelMat[i]*(OS.alphas[i]-alphaIold)*OS.K[i,j]- OS.labelMat[j]*(OS.alphas[j]-alphaJold)*OS.K[j,j]
        if (0 < OS.alphas[i] < OS.C):
            OS.b = b1
        elif (0 < OS.alphas[j]<OS.C):
            OS.b = b2
        else:
            OS.b = (b1 + b2)/2.0
        return 1
    else:
        return 0


#SMO函数，用于快速求解出alpha
def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)): #输入参数：数据特征，数据类别，参数C，阀值toler，最大迭代次数，核函数（默认线性核）
    OS = OptStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C,toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(OS.m): #遍历所有数据
                alphaPairsChanged += innerL(i, OS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)) #显示第多少次迭代，那行特征数据使alpha发生了改变，这次改变了多少次alpha
            iter += 1
        else:
            nonBoundIs = np.nonzero((OS.alphas.A > 0) * (OS.alphas.A < C))[0]
            for i in nonBoundIs: #遍历非边界的数据
                alphaPairsChanged += innerL(i, OS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iter)
    return OS.b, OS.alphas


def testRbf(data_train,data_test):
    data, label = load_dataset(data_train) #读取训练数据
    b, alphas = smoP(data, label, 200, 0.0001, 10000, ('rbf', 1.3)) 
    #通过SMO算法得到b和alpha
    datMat = np.mat(data)
    labelMat = np.mat(label).transpose()
    svInd=np.nonzero(alphas)[0]  #选取不为0数据的行数（也就是支持向量）
    sVs=datMat[svInd]  #支持向量的特征数据
    labelSV = labelMat[svInd] #支持向量的类别（1或-1）
    print("there are %d Support Vectors" % np.shape(sVs)[0])  #打印出共有多少的支持向量
    m, n = np.shape(datMat) #训练数据的行列数
    errorCount = 0
    for i in range(m):
        kernelEval = kernel(sVs, datMat[i,:],('rbf', 1.3)) #将支持向量转化为核函数
        predict=kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b  #这一行的预测结果（代码来源于《统计学习方法》p133里面最后用于预测的公式）注意最后确定的分离平面只有那些支持向量决定。
        if np.sign(predict)!=np.sign(label[i]): 
            #sign函数 -1 if x < 0,  0 if x==0, 1 if x > 0
            errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m)) #打印出错误率
    data_test,label_test = load_dataset(data_test) #读取测试数据
    errorCount_test = 0
    datMat_test = np.mat(data_test)
    labelMat = np.mat(label_test).transpose()
    m,n = np.shape(datMat_test)
    for i in range(m): #在测试数据上检验错误率
        kernelEval = kernel(sVs, datMat_test[i,:],('rbf', 1.3))
        predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
        if np.sign(predict)!=np.sign(label_test[i]):
            errorCount_test += 1
    print("the test error rate is: %f" % (float(errorCount_test)/m))


def main():
    filename1 = 'c:/users/kxshi/desktop/sampledata.txt'
    filename2 = 'c:/users/kxshi/desktop/testdata.txt'
    testRbf(filename1, filename2)

if __name__ == '__main__':
    main()
