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

from sklearn.linear_model import LinearRegression
regressor = LinearRegression() #建立LinearRegression的實體物件
regressor.fit(x_train,y_train) #答案丟入做訓練
#做資料的預測
y_pred = regressor.predict(x_test)

import statsmodels.api as sm

#X_train = np.append(arr = np.ones((40 ,1)).astype(int),values = x_train, axis = 1) #axis =0 做row|| =1做col
X_opt = x_train[:,[0,1,2,3,4,5,6,7,8,9,10]]
X_opt = np.array(X_opt,dtype=float) #做型態轉換
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit() #endog應變量 #exog自變量 載入
regressor_OLS.summary() #內建輸出資料

X_opt = x_train[:,[1,2,3,4,5,6,7,8,9,10]]
X_opt = np.array(X_opt,dtype=float) #做型態轉換
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit() #endog應變量 #exog自變量 載入
regressor_OLS.summary() #內建輸出資料

X_opt = x_train[:,[1,2,4,5,6,7,8,9,10]]
X_opt = np.array(X_opt,dtype=float) #做型態轉換
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit() #endog應變量 #exog自變量 載入
regressor_OLS.summary() #內建輸出資料

X_opt = x_train[:,[1,4,5,6,7,8,9,10]]
X_opt = np.array(X_opt,dtype=float) #做型態轉換
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit() #endog應變量 #exog自變量 載入
regressor_OLS.summary() #內建輸出資料