import numpy as np
import matplotlib.pyplot as pyplot
import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf' , random_state = 0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(x_test, y_pred)


#使用交叉驗證 讓模型更好
from sklearn.model_selection import cross_val_score
#實體化cross_val_score
'''cross_val_score裡面介紹
estimator : 把訓練好的模型實體化的物件做estimator
x,y : 針對訓練集合(x_train、y_train)來做
cv : 就是k值 default = 5
'''
#這邊在做，把資歷切成十等分and做十次  accuracies為準確率 Ex: 0.8 正確率80%
accuracies = cross_val_score(estimator = classifier, x = x_train, y = y_train, cv =10)
#mean為裡面的function 用來取平均
accuracies.mean()
#std為裡面的function 用來取標準差
accuracies.std()
#觀察平均與標準差的差距
