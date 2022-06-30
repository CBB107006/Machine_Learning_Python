import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('insurance.csv')


x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,6].values

#做Miss_Data使用_防止有nan
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy="mean",fill_value=None)
imputer = imputer.fit(x[:,0:1])
x[:,0:1] = imputer.transform(x[:,0:1])
imputer = imputer.fit(x[:,2:4])
x[:,2:4] = imputer.transform(x[:,2:4])

#標籤編碼
from sklearn.preprocessing import LabelEncoder
labelencoder_sex = LabelEncoder()
labelencoder_smoker = LabelEncoder()
labelencoder_region = LabelEncoder()

x[:,1] = labelencoder_sex.fit_transform(x[:,1])
x[:,4] = labelencoder_smoker.fit_transform(x[:,4])
x[:,5] = labelencoder_region.fit_transform(x[:,5])

#製作虛擬編碼
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
#以下的程式說明
'''
將sex與smoker與region分開做，由於做完編碼後都asign給X變數
But由於前一個做了OneHotEncoder，所以表格的索引會增加。
導致下一個表格要抓取的位置被變動了，所以得看下上一次做完OneHotEncoder後的資料
才能得知下一次要抓取的索引值為多少。
1.X asign 給 x(上一次的資料)
2.X asign 給 X 
'''

ct_sex = ColumnTransformer([("sex",OneHotEncoder(),[1])],remainder = 'passthrough')
X = ct_sex.fit_transform(x)
ct_smoker = ColumnTransformer([("smoker",OneHotEncoder(),[5])],remainder = 'passthrough')
X = ct_smoker.fit_transform(X)
ct_region = ColumnTransformer([("region",OneHotEncoder(),[7])],remainder = 'passthrough')
X = ct_region.fit_transform(X)

#分割訓練集合與測試集合
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#特徵縮放
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

