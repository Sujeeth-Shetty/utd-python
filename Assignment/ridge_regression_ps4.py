#!/usr/bin/env python
# coding: utf-8

# # Ridge regression

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-white')


class RidgeRegression:
    def __init__(self, alpha = 0.01,iterations=1000,lamda=0.5):
        self.alpha,self.iterations, self.lamda= alpha,iterations,  lamda        
    
    def closed_form_ridge_solution(self,X,y): 
    #closed form solution
        m,n = X.shape
        I = np.eye((n))
        G=self.lamda * I
        return (np.linalg.inv(X.T @ X + G) @ X.T @ y)

    def cost_function(self,X,y,beta,m):
    #Cost function for ridge regression (regularized L2)
        #Initialization
        #m = len(y) 
        J = 0

        #Vectorized implementation
        h = X @ beta
        J_reg = (self.lamda / (2*m)) * np.sum(np.square(beta))
        J = float((1./(2*m)) * (h - y).T @ (h - y)) + J_reg;
        return(J) 

    def gradient_descent(self,X,y,beta):
        #Gradient descent for ridge regression
        m = np.size(y)
        J_history = np.zeros(self.iterations)
        for i in range(self.iterations):
            #Hypothesis function
            h = np.dot(X,beta)

            #Gradient function in vectorized form
            beta = beta - self.alpha * (1/m)* (  (X.T @ (h-y)) + self.lamda * beta )

            #Cost function in vectorized form       
            J_history[i] = self.cost_function(X,y,beta,m)

        return beta ,J_history
    
    def fit(self, X, y):
        n = X.shape[1]
        #Add a column of 1
        one_column = np.ones((X.shape[0],1))
        X = np.concatenate((one_column, X), axis = 1)
        
        # initializing the parameter vector
        beta = np.zeros(n+1)
        
        #gradient descent
        beta, cost = self.gradient_descent(X,y,beta)
        return cost,beta



## Data generation
df = pd.DataFrame(np.random.normal(size=(500,4)),columns=['x1','x2','x3','y'])
df['y'] += df[['x1','x2','x3']]@np.random.uniform(size=(3,))
y=df['y']
X=df.iloc[:,0:3]


#Create Model 
learning_rate=0.001
epochs=1500
#set threshold
threshold=1
#create model
model=RidgeRegression(learning_rate,epochs,threshold)
#fit the model
cost,beta=model.fit(X,y)
print("Optimum Parameters of Ridge Regression")
print(beta)

print("Min cost :", min(cost))



#plot cost vs epcohs
import matplotlib.pyplot as plt
cost = list(cost)
n_iterations = [x for x in range(1,epochs+1)]
plt.plot(n_iterations, cost)
#plt.legend(loc='best')
plt.xlabel('No. of iterations')
plt.ylabel('Cost')


#Closed form solution 

beta_closed=model.closed_form_ridge_solution(X,y)
print("Parameters of closed form solution")
print(beta_closed)

