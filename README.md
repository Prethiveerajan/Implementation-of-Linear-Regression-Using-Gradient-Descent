# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the linear regression using gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1.  Use the standard libraries in python for Gradient Design.
2.  Upload the dataset and check any null value using .isnull() function.
3.  Declare the default values for linear regression.
4.  Calculate the loss usinng Mean Square Error.
5. Predict the value of y.
6.  Plot the graph respect to hours and scores using scatter plot function. 


## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: PrethiveeRajan P
RegisterNumber: 212221230079 
*/
```

````
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
df=pd.read_csv('student_scores - student_scores.csv')
print(df.head())
print(df.tail())

#checking for null values in dataset
print(df.isnull().sum())

#To calculate Gradient decent and Linear Descent
x=df.Hours
print(x.head())

y=df.Scores
print(y.head())

n=len(x)
m=0
c=0
L=0.001
loss=[]
for i in range(10000):
    ypred = m*x + c
    MSE = (1/n) * sum((ypred - y)*2)
    dm = (2/n) * sum(x*(ypred-y))
    dc = (2/n) * sum(ypred-y)
    c = c-L*dc
    m = m-L*dm
    loss.append(MSE)
print(m,c)

#plotting Linear Regression graph
y_pred=m*x+c
plt.scatter(x,y,color="violet")
plt.plot(x,y_pred,color="purple")
plt.xlabel("Study Hours")
plt.ylabel("Scores")
plt.title("Study hours vs Scores")
print(plt.show())

#plotting Gradient Descent graph
plt.plot(loss, color="skyblue")
plt.xlabel("Iterations")
plt.ylabel("Loss")
print(plt.show())
````

## Output:
![OUTPUT](ori.png)
![OUTPUT](ori2.png)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
