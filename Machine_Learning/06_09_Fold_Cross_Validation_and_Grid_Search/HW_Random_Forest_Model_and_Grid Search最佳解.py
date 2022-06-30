import numpy as np
import matplotlib.pyplot as pyplot
import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#n_estimators 決定有機棵樹在這個森林裡面
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state=0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm =confusion_matrix(y_test, y_pred) #正確率91%

from sklearn.model_selection import GridSearchCV

#下面是把需要的參數先設定好 parameters為一list 把C值設定為不同的值 1、10...等 and 搭配kernel
parameters = [{'n_estimators': [1, 10, 100], 'criterion': ['entropy']},
              {'n_estimators': [1, 10, 100], 'criterion': ['gini']}]
              #linear沒有gamma可以設定 而gamma為2D的半徑 gamma越大半徑越大;反之。
'''parameters參數介紹
C : 在SVM裡面為錯誤容錯的範圍 C越小容錯範圍越大;反之。 帶點有懲罰值的效果

'''
#grid_search把組合拿去跑出來
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
'''GridSearchCV參數介紹
estimator : 為所建立的模型 這邊只是把它實體化成classifier
param_grid : 為網格的參數
n_jobs : 當資料量大的時候 設定為-1會也比較好的效能去跑
'''
#對模型做fit結果
grid_search = grid_search.fit(x_train, y_train)
#grid_search裡面有很多好用的function best_score_找出最好的
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_  #跑出最好的結果
#n_estimators及criterion的最佳組合 
#n_estimators : 100
#criterion : gini