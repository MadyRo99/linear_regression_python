# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 19:00:00 2019

@author: Madalin
"""

import matplotlib.pyplot as plt
import numpy as np

#ComputeCost Function - Solves the cost for the given theta_0 and theta_1 parameters
def computeCost(X, Y, theta_0, theta_1):
    total_cost = 0
    
    #Calculate the Mean Squared Error
    for i in range(len(X)):
        total_cost += np.power((theta_0 + theta_1 * X[i] - Y[i]), 2)
        
    return (1/(2*len(X))) * total_cost

#GradientDescent Function - Finds the minimum cost for the given model and returns the most optimal theta_0 and theta_1 parameters
def gradientDescent(X, Y, theta_first, theta_second, iters, alpha):
    theta_0 = theta_first
    theta_1 = theta_second
    
    #Iterate in order to get better values for theta_0 and theta_1
    for i in range(iters):
        theta_0_partial = 0
        theta_1_partial = 0
        
        #Calculate the sum of the derivatives, both for theta_0 and theta_1
        for i in range(len(X)):
            theta_0_partial += theta_0 + theta_1 * X[i] - Y[i]
            theta_1_partial += (theta_0 + theta_1 * X[i] - Y[i]) * X[i]
            
        #Assign a new value for theta_0 and theta_1, by adding/substracting from the actual theta values,
        #depending on the sign of the slope (positive/negative) and multiplying by the learning rate alpha
        theta_0 = theta_0 - alpha * (1/len(X)) * theta_0_partial
        theta_1 = theta_1 - alpha * (1/len(X)) * theta_1_partial
        
    cost = computeCost(X, Y, theta_0, theta_1)
    
    return (theta_0, theta_1, cost)

#Read data from DB
my_data = np.genfromtxt('data.csv', delimiter=',')

#Assign data values to X and Y
X = my_data[:, 0].reshape(-1, 1)
Y = my_data[:, 1].reshape(-1, 1)

#Set a starting point for theta_0 and theta_1 parameters
theta_0 = 14570
theta_1 = 11115

#Set an alpha learning rate and the number of iterations
alpha = 0.0001
iters = 10000

#Use the Gradient Descent function to get the best theta_0 and theta_1 values for the minimum cost
theta_0, theta_1, cost = gradientDescent(X, Y, theta_0, theta_1, iters, alpha)

print("Valorile pentru theta_0 si theta_1 sunt:")
print(theta_0, theta_1)
print("Costul final este:")
print(cost)

#Plot the data and the regression line
plt.scatter(X, Y)
axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals = theta_0 + theta_1 * x_vals
plt.plot(x_vals, y_vals, '--')   
print()