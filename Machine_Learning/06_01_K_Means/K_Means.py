import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Mall_Customers.csv')

x = dataset.iloc[:,3:5].values

#以下是在做手肘法則 查看我們要分幾群比較適當

from sklearn.cluster import KMeans
wcss = []
#利用for迴圈把每一次不同的群(1群、2群、3群、4群...10群)畫出來得結果怎麼樣
for i in range(1,11): #觀察1-10
    kmeans = KMeans(n_clusters = i,max_iter = 300, n_init = 10,init = 'k-means++',random_state = 0)
    '''
    n_clusters : 要分群的數量
    init : 初始化中心點
    n_init : K-means演算法執行幾次不同中心點(隨機)的選擇 把最好的選出來
    max_iter : 最大疊代次數
    '''
    kmeans.fit(x) #將切個的資料丟入模型做訓練
    wcss.append(kmeans.inertia_) #inertia_ : 組內平方和 透過這個屬性可以取出來(每一次訓練出來) 在append到WCSS
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters') #x軸要分群的數量
plt.ylabel('WCSS')#y軸組內平方和
plt.show #得知K=5 則分五群最理想(對這個案例)

#分5群訓練
kmeans = KMeans(n_clusters = 5,max_iter = 300, n_init = 10,init = 'k-means++',random_state = 0)
y_kmeans = kmeans.fit_predict(x) #內建有fit_predict的function可以用
'''結果讀作 編號0顧客分在第4群
           編號1顧客分在第3群
           編號2顧客分在第4群
'''
#透過圖形觀看分層結果
plt.scatter(x[y_kmeans == 0, 0],x[y_kmeans == 0, 1],s = 100, c = 'red', label = 'Cluster0')
'''
讀作
x[y_kmeans == 0, 0],x[y_kmeans == 0, 1] : 是x、y的座標
    x[y_kmeans == 0, 0] : x裡面的列 只要y_kmeans等於0的這一列的索引值 對應到x裡面第0個col 把值取出 例 y_kmeans = 150列 在x第150索引值的第0個col把78這個值取出來
    x[y_kmeans == 0, 1] : x裡面的列 只要y_kmeans等於1的這一列的索引值 對應到x裡面第1個col 把值取出 例 y_kmeans = 150列 在x第150索引值的第1個col把17這個值取出來
s : 點的大小
label : 識別的標籤
'''
plt.scatter(x[y_kmeans == 1, 0],x[y_kmeans == 1, 1],s = 100, c = 'blue', label = 'Cluster1')
plt.scatter(x[y_kmeans == 2, 0],x[y_kmeans == 2, 1],s = 100, c = 'green', label = 'Cluster2')
plt.scatter(x[y_kmeans == 3, 0],x[y_kmeans == 3, 1],s = 100, c = 'cyan', label = 'Cluster3')
plt.scatter(x[y_kmeans == 4, 0],x[y_kmeans == 4, 1],s = 100, c = 'magenta', label = 'Cluster4')
#K-Means底下好用的屬性「cluster_centers_」會把中心點取出來 為一個陣列 把分群的中心點儲存起來
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300, c = 'yellow', label = 'Centroids')#每一群的中心點
'''
kmeans.cluster_centers_[:,0]當作x 
    kmeans.cluster_centers_[:,0] 讀作 所有列的第0個col當作x
kmeans.cluster_centers_[:,1]當作y
    kmeans.cluster_centers_[:,1] 讀作 所有列的第1個col當作y
'''
plt.title('Clusters of clients')
plt.xlabel('Annual Income (k$)')#收入
plt.ylabel('Spending Score (1-100)')#消費指數
plt.legend()
plt.show

#主力目標 右上角綠色部分 收入高消費高