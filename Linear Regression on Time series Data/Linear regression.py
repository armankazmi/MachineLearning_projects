#Importing Modules
import pandas as pd
from matplotlib import pyplot as plt
import datetime as dt
import numpy as np

#Importing the dataset
data = pd.read_csv("train.csv",parse_dates = ['id'])


#Creating Features for regression by using year and month independently as features
data['year'] = data['id'].dt.year
data['month'] = data['id'].dt.month

#Defining Our Features
column = ['id','value']
X = data.drop(column, axis = 1)
y = data.value


#Splitting The dataset for training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20)


#Applying Linear Regression by introducing Polynomial Features
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
poly_model = make_pipeline(PolynomialFeatures(10),LinearRegression())
poly_model.fit(np.asfarray(X_train), y_train)
prediction = poly_model.predict(np.asfarray(X_test))


#Measuring the scores
from sklearn.metrics import mean_squared_error,r2_score
test_rmse = np.sqrt(mean_squared_error(y_test, prediction))
test_r2 = r2_score(y_test, prediction)

print("The root mean squared error is : %8f"%(test_rmse))
print("The r2 score is : %8f"%(test_r2))
#test_rmse = 3.2109132369315385
#r2_score = 0.8866454336586124
