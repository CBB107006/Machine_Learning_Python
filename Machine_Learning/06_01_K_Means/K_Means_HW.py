import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Mall_Customers.csv')

x = dataset.iloc[:,3:5].values

from sklearn.cluster import KMeans
wcss = []

#分4群
kmeans = KMeans(n_clusters = 4,max_iter = 300, n_init = 10,init = 'k-means++',random_state = 0)
y_kmeans = kmeans.fit_predict(x)

plt.scatter(x[y_kmeans == 0, 0],x[y_kmeans == 0, 1],s = 100, c = 'red', label = 'Cluster0')
plt.scatter(x[y_kmeans == 1, 0],x[y_kmeans == 1, 1],s = 100, c = 'blue', label = 'Cluster1')
plt.scatter(x[y_kmeans == 2, 0],x[y_kmeans == 2, 1],s = 100, c = 'green', label = 'Cluster2')
plt.scatter(x[y_kmeans == 3, 0],x[y_kmeans == 3, 1],s = 100, c = 'cyan', label = 'Cluster3')
#K-Means底下好用的屬性「cluster_centers_」會把中心點取出來 為一個陣列 把分群的中心點儲存起來
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300, c = 'yellow', label = 'Centroids')#每一群的中心點
plt.title('Clusters of clients')
plt.xlabel('Annual Income (k$)')#收入
plt.ylabel('Spending Score (1-100)')#消費指數
plt.legend()
plt.show

#分5群
kmeans = KMeans(n_clusters = 5,max_iter = 300, n_init = 10,init = 'k-means++',random_state = 0)
y_kmeans = kmeans.fit_predict(x)

plt.scatter(x[y_kmeans == 0, 0],x[y_kmeans == 0, 1],s = 100, c = 'red', label = 'Cluster0')
plt.scatter(x[y_kmeans == 1, 0],x[y_kmeans == 1, 1],s = 100, c = 'blue', label = 'Cluster1')
plt.scatter(x[y_kmeans == 2, 0],x[y_kmeans == 2, 1],s = 100, c = 'green', label = 'Cluster2')
plt.scatter(x[y_kmeans == 3, 0],x[y_kmeans == 3, 1],s = 100, c = 'cyan', label = 'Cluster3')
plt.scatter(x[y_kmeans == 4, 0],x[y_kmeans == 4, 1],s = 100, c = 'magenta', label = 'Cluster4')
#K-Means底下好用的屬性「cluster_centers_」會把中心點取出來 為一個陣列 把分群的中心點儲存起來
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300, c = 'yellow', label = 'Centroids')#每一群的中心點
plt.title('Clusters of clients')
plt.xlabel('Annual Income (k$)')#收入
plt.ylabel('Spending Score (1-100)')#消費指數
plt.legend()
plt.show

#分6群
kmeans = KMeans(n_clusters = 6,max_iter = 300, n_init = 10,init = 'k-means++',random_state = 0)
y_kmeans = kmeans.fit_predict(x)

plt.scatter(x[y_kmeans == 0, 0],x[y_kmeans == 0, 1],s = 100, c = 'red', label = 'Cluster0')
plt.scatter(x[y_kmeans == 1, 0],x[y_kmeans == 1, 1],s = 100, c = 'blue', label = 'Cluster1')
plt.scatter(x[y_kmeans == 2, 0],x[y_kmeans == 2, 1],s = 100, c = 'green', label = 'Cluster2')
plt.scatter(x[y_kmeans == 3, 0],x[y_kmeans == 3, 1],s = 100, c = 'cyan', label = 'Cluster3')
plt.scatter(x[y_kmeans == 4, 0],x[y_kmeans == 4, 1],s = 100, c = 'magenta', label = 'Cluster4')
plt.scatter(x[y_kmeans == 5, 0],x[y_kmeans == 5, 1],s = 100, c = 'black', label = 'Cluster5')
#K-Means底下好用的屬性「cluster_centers_」會把中心點取出來 為一個陣列 把分群的中心點儲存起來
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300, c = 'yellow', label = 'Centroids')#每一群的中心點
plt.title('Clusters of clients')
plt.xlabel('Annual Income (k$)')#收入
plt.ylabel('Spending Score (1-100)')#消費指數
plt.legend()
plt.show