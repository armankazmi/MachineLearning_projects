#Importing Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix 

#data
data = pd.read_csv("17054.csv")

#X and y for classification
X = data.drop('label',axis =1)
y = data.label

#Scaling the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

#Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=51)


#model without Grid search
svm_model = svm.SVC(kernel='rbf',gamma='scale', C=1) #C value varied
svm_model.fit(X_train,y_train)
predictions = svm_model.predict(X_test)
#score = accuracy_score(predictions, y_test)
print("The report without using Grid search CV : ")
print(classification_report(y_test, predictions))


#Grid Search for best parameters
from sklearn.model_selection import GridSearchCV 
param_grid = {'C': [0.01,0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 'auto','scale'], 
              'kernel': ['rbf']}  
  
grid = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 3) 
  
# fitting the model for grid search 
grid.fit(X_train, y_train)
grid_predictions = grid.predict(X_test)
print("After using Grid Search CV: ")
print(classification_report(y_test, grid_predictions)) 
