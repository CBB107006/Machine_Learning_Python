import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('mnist_train.csv')

x = dataset.iloc[:,1:].values
y = dataset.iloc[:,0].values


show_img = np.reshape(x[188,:],(28,28))#把索引值第188列的所有行抓出來呈現
plt.matshow(show_img,cmap = plt.get_cmap('gray'))
plt.show

x[x>0]=1 #讀作if x 自變量裡面 它的value若大於1就設定為1小於1為0 將圖片設定成黑白


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.svm import SVC  #模型訓練
classifier = SVC(kernel = 'rbf',random_state = 0) #Kernel使用rbf高斯和函數
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix #做正確錯誤對照表
cm =confusion_matrix(y_test, y_pred)

#跑出準確率的結果
from sklearn.metrics import accuracy_score 
accuracy_score(y_test, y_pred)

import matplotlib.image as mpimg
img = mpimg.imread('4.png') #讀取圖片存取在 會是28*28像素 (28,28,3) 
                            #多一個3的維度是因為電腦認定它為彩色圖片
                            #所以會有3個channel
#先做一個前處理 取出我們要的28*28就好
img = img[:,:,2]
plt.matshow(img,camp = plt.get_camp('gray'))
plt.show()
test_img = img.reshape((1,784))

img_class = classifier.predict(test_img)
