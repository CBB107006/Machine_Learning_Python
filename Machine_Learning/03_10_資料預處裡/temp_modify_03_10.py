import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#-------------------------------------
np1 = np.array([1,2,3,4,5,6])
np1 = np1.reshape([2,3])
print(np1.ndim,np1.shape,np1.dtype)
#-------------------------------------
np1 = np1.astype('int32')
np1.dtype


#-------------------------------------

dataset = pd.read_csv('Data.csv')

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values
#-------------------------------------03_10

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan,strategy="mean",fill_value=None)
imputer = imputer.fit(x[:,1:3])#索引值1~3不包含3 改變x自變數
x[:,1:3] = imputer.transform(x[:,1:3])


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_x = LabelEncoder()
x[:, 0] =labelencoder_x.fit_transform(x[:, 0])


from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("Country",OneHotEncoder(),[0])],remainder = 'passthrough')
#將Country做虛擬編碼 索引值0的地方
X = ct.fit_transform(x)

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
