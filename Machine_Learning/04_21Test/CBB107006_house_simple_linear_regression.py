import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('house.csv')

x = dataset.iloc[:,[7]].values
y = dataset.iloc[:,[8]].values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan,strategy="mean",fill_value=None)
imputer = imputer.fit(x[:,:])#索引值1~3不包含3 改變x自變數
x[:,:] = imputer.transform(x[:,:])


imputer = SimpleImputer(missing_values=np.nan,strategy="mean",fill_value=None)
imputer = imputer.fit(y[:,:])#索引值1~3不包含3 改變x自變數
y[:,:] = imputer.transform(y[:,:])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression() #建立LinearRegression的實體物件
regressor.fit(x_train,y_train) #答案丟入做訓練

#做資料的預測
y_pred = regressor.predict(x_test) #x_test丟入做預測;y_pred為預測結果;y_test為正確解答

#做資料視覺化處理train
plt.scatter(x_train,y_train,color='red') #將x_train y_train 資料弄到圖中 散點圖
plt.plot(x_train,regressor.predict(x_train),color = 'blue')  #畫折線圖
plt.title('median_income and median_house_value (training set')
plt.xlabel('median_income')
plt.ylabel('median_house_value')
plt.show()

#做資料視覺化處理test
plt.scatter(x_test,y_test,color='red') #將x_train y_train 資料弄到圖中
plt.plot(x_train,regressor.predict(x_train),color = 'blue')
plt.title('median_income and median_house_value (testing set')
plt.xlabel('median_income')
plt.ylabel('median_house_value')
plt.show()