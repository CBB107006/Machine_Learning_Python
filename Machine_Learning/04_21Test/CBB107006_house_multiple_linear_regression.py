
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('house.csv')

x = dataset.iloc[:,[2,3,4,5,6,7,9]].values
y = dataset.iloc[:,[8]].values

#Missing Data 處理
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan,strategy="mean",fill_value=None)
imputer = imputer.fit(x[:,0:6])#索引值1~3不包含3 改變x自變數
x[:,0:6] = imputer.transform(x[:,0:6])

imputer = SimpleImputer(missing_values=np.nan,strategy="most_frequent",fill_value=None)
imputer = imputer.fit(x[:,[6]])#索引值1~3不包含3 改變x自變數
x[:,[6]] = imputer.transform(x[:,[6]])

imputer = SimpleImputer(missing_values=np.nan,strategy="mean",fill_value=None)
imputer = imputer.fit(y[:,:])#索引值1~3不包含3 改變x自變數
y[:,:] = imputer.transform(y[:,:])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_x = LabelEncoder()
x[:,6] =labelencoder_x.fit_transform(x[:, 6])

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("ocean_proximity",OneHotEncoder(),[6])],remainder = 'passthrough')
#將State做虛擬編碼 索引值0的地方
X = ct.fit_transform(x)

#做虛擬變量陷阱處理
X = X[:,1:]#把索引值0的去除 要去除一個

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression() #建立LinearRegression的實體物件
regressor.fit(x_train,y_train) #答案丟入做訓練
#做資料的預測
y_pred = regressor.predict(x_test)


#載入套件做backword  Elimination
import statsmodels.api as sm

X_train = np.append(arr = np.ones((16512 ,1)).astype(int),values = x_train, axis = 1) #axis =0 做row|| =1做col
X_opt = X_train[:,[0,1,2,3,4,5,6,7,8,9,10]]
X_opt = np.array(X_opt,dtype=float) #做型態轉換
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit() #endog應變量 #exog自變量 載入
regressor_OLS.summary() #內建輸出資料