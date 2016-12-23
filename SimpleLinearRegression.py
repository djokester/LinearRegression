#required import statements 
import numpy as np
import matplotlib.pyplot as plt
import functools as fn
import random 
import pandas as pd
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

#computing the cost for a particular value of theta
def ComputeCost(X, Y, theta):
    return(np.sum(np.square(np.dot(theta, X)-Y))/(2*len(Y)))
    
#(applyin the gradient descent 
def gradient_descent(X, y, iterations, theta, alpha):
    J_log = []
    for i in range(0, iterations):
        cost = ComputeCost(X, y, theta)
        J_log.append(cost)
        theta=theta - alpha/len(y)*(np.dot((np.dot(theta,X)-y),X.transpose()))
    return (theta,J_log)          
    
#import the data sets
f = open('ex1data1.txt').read() #chose the file and the location
f = f.split()
x = []
y = []
for line in f:
    data = str(line).split(',')
    x.append(float(data[0]))
    y.append(float(data[1]))

print(x)
print(y)


#defining the variables that we shall work with
xo = []
for i in range(0, len(y)):
    xo.append(1) 

X = np.matrix([xo, x])
Y = y
theta = np.ones([1, 2])

#fine tune these two values to match your needs.
iterations = 10000
alpha = 0.02

#Applying Gradient Descent and Extracting the Cost and the Optimised Theta Values
a, b  = gradient_descent(X,Y,iterations, theta,alpha)
theta_dash = a
print(theta_dash)

y_dash = np.dot(theta_dash,X) 
print(y_dash)
#Plotting the Regression Model
plt.plot(X[1,:] ,y_dash, 'ro')
#Plotting the Training set data as a scatter plot 
plt.scatter(X[1,:],y)

plt.show()