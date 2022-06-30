import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('wine.csv')

x = dataset.iloc[:,0:11].values
y = dataset.iloc[:,[11]].values


#Missing Data 處理
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy="mean",fill_value=None)
imputer = imputer.fit(x[:,:])#索引值1~3不包含3 改變x自變數
x[:,:] = imputer.transform(x[:,:])

imputer = SimpleImputer(missing_values=np.nan,strategy="most_frequent",fill_value=None)
imputer = imputer.fit(y[:,:])#索引值1~3不包含3 改變x自變數
y[:,:] = imputer.transform(y[:,:])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

#丟入邏輯回規模型訓練
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state= 0)
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)


#作資料核對
from sklearn.metrics import confusion_matrix
cm =confusion_matrix(y_test, y_pred)
###         cm圖片:左上到右下(對角)預測正確的    ###
###         (0,1)讀作正確為0 預測為1            ###
###         (1,0)讀作正確為1 預測為0            ###



