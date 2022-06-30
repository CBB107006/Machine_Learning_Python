import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Iris.csv')

x = dataset.iloc[:,1:5].values
y = dataset.iloc[:,[5]].values

#Missing Data 處理
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan,strategy="mean",fill_value=None)
imputer = imputer.fit(x[:,0:4])#索引值1~4不包含4 改變x自變數
x[:,0:4] = imputer.transform(x[:,0:4])

imputer = SimpleImputer(missing_values=np.nan,strategy="most_frequent",fill_value=None)
imputer = imputer.fit(y[:,[0]])#索引值1~3不包含3 改變x自變數
y[:,[0]] = imputer.transform(y[:,[0]])

#資料切割 分訓練及合測試集合
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#做標準化
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

#import SVC 將訓練集合帶入訓練模型
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear',random_state = 0)
classifier.fit(x_train,y_train)

#測試集合帶入預測
y_pred = classifier.predict(x_test)

#作資料核對
from sklearn.metrics import confusion_matrix
cm =confusion_matrix(y_test, y_pred)