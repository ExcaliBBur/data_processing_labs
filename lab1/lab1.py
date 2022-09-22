import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

# first task Kmeans++
X, y = make_blobs(random_state=1, n_samples=300, centers=6) # создаём 300 точек с 6 кластерами
criteries = []
for k in range(2,10): #ищем оптимальное кол-во кластером для Kmeans методом локтя
    kmeans_model = KMeans(n_clusters = k, random_state=3)
    kmeans_model.fit(X)
    criteries.append(kmeans_model.inertia_)

# plt.plot(range(2,10), criteries) оптимальное кол-во кластеров = 5
# plt.show()
kmeansModel=KMeans(n_clusters=5, random_state=0)
kmeansModel.fit(X)
labels = kmeansModel.labels_
plt.scatter(X[:,0], X[:,1], c=labels)
plt.title("Kmeans for random data")
plt.show()


# first task DBSCAN
clustering = DBSCAN(eps = 1, min_samples = 10).fit_predict(X) #подбираем оптимальные параметры
plt.scatter(X[:,0], X[:,1], c=clustering)
plt.title("DBSCAN for random data")
plt.show()

# second task 

#Kmeans
data = pd.read_csv("Mall_Customers.csv")
X = data[['Spending Score (1-100)', 'Annual Income (k$)']].iloc[:, :].values
criteries = []
for k in range(2,10): #ищем оптимальное кол-во кластером для Kmeans методом локтя
    kmeans_model = KMeans(n_clusters = k, random_state=3)
    kmeans_model.fit(X)
    criteries.append(kmeans_model.inertia_)
# plt.plot(range(2,10), criteries) #оптимальное кол-во кластеров = 6
# plt.show()
kmeansModel=KMeans(n_clusters=6, random_state=0)
kmeansModel.fit(X)
labels = kmeansModel.labels_
plt.scatter(X[:,0], X[:,1], c=labels)
plt.title("Kmeans for score(y) and income(x)")
plt.show()

#DBSCAN
clustering = DBSCAN(eps = 8, min_samples = 2).fit_predict(X) #подбираем оптимальные параметры 6 10
plt.scatter(X[:,0], X[:,1], c=clustering)
plt.title("DBSCAN for score(y) and income(x)")
plt.show()