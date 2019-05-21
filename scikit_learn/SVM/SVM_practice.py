# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 13:40:42 2018

@author: kxshi
"""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import  train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report,  confusion_matrix

plt.style.use('seaborn')

# 人脸识别案例，带标记的人脸
# 1提取数据
faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)  # len=8
print(faces.images.shape, faces.data.shape, faces.target.shape, sep='\t')
# (1348, 62, 47) /  (1348, 2914) / (1348,)

# 2预处理器和分类器打包成管道
pca = PCA(n_components=150, whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced', gamma='auto')
model = make_pipeline(pca, svc)

# 3数据集分解
xtr, xte, ytr, yte = train_test_split(faces.data, faces.target, test_size=0.2, random_state=42)
print(xtr.shape, xte.shape) #  (1011, 2914) (337, 2914)  default 25%

# 4网络搜索交叉检验寻找最优参数，构造最优模型
tune_params = {'svc__C': [0.1, 0.5, 1, 2, 3, 4],   # 边界硬度
          'svc__gamma': [2e-3, 3e-3, 4e-3]}  # 核函数大小
grid = GridSearchCV(model, tune_params, cv=5, n_jobs=4, verbose=1, iid=False).fit(xtr, ytr)
print(grid.best_params_) # 最优参数 {'svc__C': 3, 'svc__gamma': 0.003}
optimal = grid.best_estimator_  # this is present only if refit is specified （default refit=True）
ypred = optimal.predict(xte)

# 5可视化部分样本对比结果
fig, ax = plt.subplots(8, 8, figsize=(12, 8))
fig.subplots_adjust(hspace=0.05, wspace=0.05)
for i, axi in enumerate(ax.flat):
    axi.imshow(xte[i].reshape(62, 47), cmap='gray', interpolation='nearest')
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel(faces.target_names[ypred[i]].split()[-1], color='g' if ypred[i]==yte[i] else 'r')
plt.suptitle('Correct labels(green)/Incorrect labels(Red)', size=18, color='r')

# 6分析测试结果
print(classification_report(yte, ypred, target_names=faces.target_names))
mat = confusion_matrix(yte, ypred)
plt.figure(figsize=(12, 8))
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False, annot_kws=dict(size=12),
            xticklabels=faces.target_names, yticklabels=faces.target_names)
plt.xlabel('Predicted', fontdict=dict(size=16,color='r'))
plt.ylabel('True', fontdict=dict(size=16,color='r'))

plt.show()