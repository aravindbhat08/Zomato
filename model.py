import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('zomato_app.csv')
x = df.iloc[:,[0,1,3,4,5,6,7,8]]
print(x)
y = df['rate']


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,random_state=10)

#Preparing Linear Regression model:
ET_Model=ExtraTreesRegressor(n_estimators=120)
ET_Model.fit(x_train,y_train)

y_predict=ET_Model.predict(x_test)

import pickle
pickle.dump(ET_Model, open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
print(y_predict)