import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values


#Missing Data 處理
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan,strategy="mean",fill_value=None)
imputer = imputer.fit(x[:,0:1])#索引值1~3不包含3 改變x自變數
x[:,0:1] = imputer.transform(x[:,0:1])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#做預測

from sklearn.linear_model import LinearRegression
regressor = LinearRegression() #建立LinearRegression的實體物件
regressor.fit(x_train,y_train) #答案丟入做訓練

#做資料的預測
y_pred = regressor.predict(x_test) #x_test丟入做預測;y_pred為預測結果;y_test為正確解答


#做資料視覺化處理train
plt.scatter(x_train,y_train,color='red') #將x_train y_train 資料弄到圖中 散點圖
plt.plot(x_train,regressor.predict(x_train),color = 'blue')  #畫折線圖
plt.title('Salary VS Experience (training set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#做資料視覺化處理test
plt.scatter(x_test,y_test,color='red') #將x_train y_train 資料弄到圖中
plt.plot(x_train,regressor.predict(x_train),color = 'blue')
plt.title('Salary VS Experience (training set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()