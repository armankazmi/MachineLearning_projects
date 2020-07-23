#author: Arman kazmi
#Curve fitting from Scratch on 110 data points
#Visualization of how the curve fits the exact data points when degree of equation is increased and overfitting starts.

#Importing Modules
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


#Importing data
data = pd.read_csv("Gaussian_noise_1.csv",header = None, names = ['X','t'])
y = np.asfarray(data['t'])
X = np.asfarray(data['X'])

#Splitting the dataset
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25)


#Shapes of X_train and X_test
m = X_train.shape
n = X_test.shape

#Defining Hypothesis function and loss function

def h(X, theta):
    return X @ theta
    #Loss function
def J(theta, X, y):
    return np.mean(np.square(h(X, theta) - y))

#Fitting the curve upto degree 15 for better visualization how overfitting starts when degree is increased
#Also the degree can be increased equal to the number od data points such that no.of degree in equation = no.of data points

degree = []
for i in range(1,15):
    degree.append(i)

    
#Curve fitting using matrix multiplication from scratch
residuals = []
param_theta = []
for deg in degree:
    x = np.zeros(shape=(m[0],deg+1))                                                 #Initializing a vector of (shape of X_train * degree+1) size
    for i in range(0, m[0]):
        for j in range(0, deg+1):
            x[i][j] = np.power(np.asfarray(X_train[i]), j)                          #This line creates a vector of the x values to the required degree
    x[:, 1:] = (x[:, 1:] - np.mean(x[:, 1:], axis=0)) / np.std(x[:, 1:], axis=0)    #Since we have added extra features to x normalizing becomes important
    theta = np.random.random(deg+1)                                                 #Initializing random weights
    losses = []                         
    #alpha = 0.01
    #for _ in range(1000):                                                          # 1000 iterations to compute the least loss                                                          
        #theta = theta - alpha * (1/m) * (x.T @ ((x @ theta) - y))                  #gradient descent method
    theta = np.linalg.pinv(x.T @ x) @ x.T @ y_train                                 #computing using normal equation method
    losses.append(J(theta, x, y_train))
    param_theta.append(theta)                                                       #Appending the parameters of weights for each degree
    predictions = h(x, theta)                                                       #predicted values for plot
    residuals.append((np.mean(predictions-y_train))/(20-deg+1))                     #goodness of fit by calculating residuals


            #Plot for visualizing how the curve fits the exact data when degree of curve is increased
    
    plt.title('degree of polynomial = '+str(deg))                 
    plt.xlabel('X values')
    plt.ylabel('targer values')
    plt.scatter(x[:, 1], predictions, label='predictions')
    plt.plot(x[:, 1], y_train, 'rx', label='original')
    plt.legend()
    plt.show()



#Testing the degree of equations on Test data and goodness of fit using R2 score and MSE
rmse = []
score = []
for deg,k in zip(degree,param_theta):
    x = np.zeros(shape=(n[0],deg+1))   
    for i in range(0, n[0]):
        for j in range(0, deg+1):
            x[i][j] = np.power(np.asfarray(X_test[i]), j)     
    x[:, 1:] = (x[:, 1:] - np.mean(x[:, 1:], axis=0)) / np.std(x[:, 1:], axis=0)
    predictions = h(x, k)
    rmse.append(J(k, x, y_test))
    score.append(r2_score(y_test, predictions, sample_weight=None, multioutput='uniform_average'))


#At degree 8 the curve fits best and provides good result
#Parameters of degree 8 equation are:-
parameter = param_theta[6]
print(parameter)

#Now Lasso or Ridge Regulrisation can be implemented to fine-tune the model obtained.
