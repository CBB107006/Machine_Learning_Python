import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,0:1].values
y = dataset.iloc[:,1].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression() #建立LinearRegression的實體物件
lin_reg.fit(X ,y)


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 6) #增廣至六次方項次 
#poly_reg = PolynomialFeatures(degree = 2) #增廣至平方項次 if degree=3 則增廣至3次方向  
''' degree越大描出來的現越接近點'''
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly ,y) 


X_grid = np.arange(min(X),max(X),0.1)#點與點之間間隔0.1
X_grid = X_grid.reshape(len(X_grid),1) #要讓X_grid變成一個col

plt.scatter(X ,y,color = 'red')
plt.plot(X_grid,lin_reg_2.predict(poly_reg.fit_transform(X_grid)),color='blue')
plt.title('Truth or Bluff(Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

predict_y_2 = lin_reg_2.predict(X_poly)