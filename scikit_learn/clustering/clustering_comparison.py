# -*- coding: utf-8 -*-

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, FeatureAgglomeration, Birch
from sklearn.cluster import DBSCAN, SpectralClustering, MeanShift, AffinityPropagation
import matplotlib.pyplot as plt
import seaborn as sns

x, y = make_blobs(n_samples=300, n_features=2, centers=4, cluster_std=0.8, shuffle=True, random_state=1)
x1, y1 = make_blobs(n_samples=500, n_features=15, centers=4, cluster_std=1.2, shuffle=True, random_state=1)

sns.set()
plt.figure(figsize=(12, 8))
plt.scatter(x[:,0], x[:,1], c=y, cmap='viridis')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Clustering', fontdict=dict(size=20, color='r'))

model1 = KMeans(n_clusters=4, max_iter=300, tol=0.0001, verbose=0, random_state=1, n_jobs=4, algorithm='auto')
model1.fit(x)
ypred1 = model1.predict(x)
print('KMeans:')
print(model1.cluster_centers_)
plt.figure(figsize=(12, 8))
plt.scatter(x[:,0], x[:,1], c=ypred1, cmap='plasma')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('KMeans', fontdict=dict(size=20, color='r'))


model2 = MiniBatchKMeans(n_clusters=4, batch_size=100, random_state=1)
model2.fit(x)
ypred2 = model2.predict(x)
print('\nMiniBatchKMeans:')
print(model2.cluster_centers_)
plt.figure(figsize=(12, 8))
plt.scatter(x[:,0], x[:,1], c=ypred2, cmap='inferno')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('MiniBatchKMeans', fontdict=dict(size=20, color='r'))

model3 = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='average')  # complete/single
ypred3 = model3.fit_predict(x)
plt.figure(figsize=(12, 8))
plt.scatter(x[:,0], x[:,1], c=ypred3, cmap='magma')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('AgglomerativeClustering', fontdict=dict(size=20, color='r'))

model4 = Birch(threshold=0.5, branching_factor=50, n_clusters=4)
model4.fit(x)
print('\nBirch:')
print(model4.subcluster_centers_.shape)
ypred4 = model4.predict(x)
plt.figure(figsize=(12, 8))
plt.scatter(x[:,0], x[:,1], c=ypred4, cmap='Spectral')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Hierarchical_Birch', fontdict=dict(size=20, color='r'))

model5 = FeatureAgglomeration(n_clusters=2, affinity='euclidean', linkage='complete')  # dimensionality reduction
x_new = model5.fit_transform(x1)
model6 = KMeans(n_clusters=4, max_iter=300, tol=0.0001, verbose=0, random_state=1, n_jobs=4)
ypred6 = model6.fit_predict(x_new)
plt.figure(figsize=(12, 8))
plt.scatter(x_new[:, 0], x_new[:, 1], c=ypred6, cmap='coolwarm')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('FeatureAgglomeration_KMeans', fontdict=dict(size=20, color='r'))

model7 = DBSCAN(eps=0.8, min_samples=5, metric='euclidean', leaf_size=30, n_jobs=4)
ypred7 = model7.fit_predict(x)
plt.figure(figsize=(12, 8))
plt.scatter(x[:, 0], x[:, 1], c=ypred7, cmap='seismic')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('DBSCAN', fontdict=dict(size=20, color='r'))

model8 = MeanShift(bandwidth=None, seeds=None, bin_seeding=False, min_bin_freq=1, cluster_all=True, n_jobs=4)
model8.fit(x)
print('\nMeanShift:')
print(model8.cluster_centers_)
ypred8 = model8.predict(x)
plt.figure(figsize=(12, 8))
plt.scatter(x[:, 0], x[:, 1], c=ypred8, cmap='jet')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('MeanShift', fontdict=dict(size=20, color='r'))

model9 = SpectralClustering(n_clusters=4, eigen_solver=None, random_state=1, gamma=1, affinity='rbf',
                            n_neighbors=10)
ypred9 = model9.fit_predict(x)
plt.figure(figsize=(12, 8))
plt.scatter(x[:, 0], x[:, 1], c=ypred9, cmap='rainbow')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('SpectralClustering', fontdict=dict(size=20, color='r'))

model10 = AffinityPropagation(damping=.8)
model10.fit(x)
print('\nAffinityPropagation:')
print(model10.cluster_centers_)
ypred10 = model10.predict(x)
plt.figure(figsize=(12, 8))
plt.scatter(x[:, 0], x[:, 1], c=ypred10, cmap='brg')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('AffinityPropagation', fontdict=dict(size=20, color='r'))

plt.show()