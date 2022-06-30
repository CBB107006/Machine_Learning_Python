
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

#Missing Data 處理
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan,strategy="mean",fill_value=None)
imputer = imputer.fit(x[:,0:3])#索引值1~3不包含3 改變x自變數
x[:,0:3] = imputer.transform(x[:,0:3])

imputer = SimpleImputer(missing_values=np.nan,strategy="most_frequent",fill_value=None)
imputer = imputer.fit(x[:,[3]])#索引值1~3不包含3 改變x自變數
x[:,[3]] = imputer.transform(x[:,[3]])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_x = LabelEncoder()
x[:, 3] =labelencoder_x.fit_transform(x[:, 3])

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("State",OneHotEncoder(),[3])],remainder = 'passthrough')
#將State做虛擬編碼 索引值0的地方
X = ct.fit_transform(x)

#做虛擬變量陷阱處理
X = X[:,1:]#把索引值0的去除 要去除一個

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression() #建立LinearRegression的實體物件
regressor.fit(x_train,y_train) #答案丟入做訓練
#做資料的預測
y_pred = regressor.predict(x_test)

#載入套件做backword  Elimination
import statsmodels.api as sm

X_train = np.append(arr = np.ones((40 ,1)).astype(int),values = x_train, axis = 1) #axis =0 做row|| =1做col
X_opt = X_train[:,[0,1,2,3,4,5]]
X_opt = np.array(X_opt,dtype=float) #做型態轉換
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit() #endog應變量 #exog自變量 載入
regressor_OLS.summary() #內建輸出資料

X_opt = X_train[:,[0,1,3,4,5]]
X_opt = np.array(X_opt,dtype=float)
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X_train[:,[0,3,4,5]]
X_opt = np.array(X_opt,dtype=float)
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X_train[:,[0,3,5]]
X_opt = np.array(X_opt,dtype=float)
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X_train[:,[0,3]]
X_opt = np.array(X_opt,dtype=float)
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()