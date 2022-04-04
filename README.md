# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the linear regression using gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. 1.Use the standard libraries in python for finding linear regression.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: PrethiveeRajan P
RegisterNumber: 212221230079 
*/
```

````
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
dataset = pd.read_csv('/content/student_scores - student_scores.csv')
dataset.head()
dataset.tail()
x  = dataset.iloc[:,:-1].values 
y  = dataset.iloc[:,1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 1/3,random_state=0)
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
plt.scatter(x_train,y_train,color = "green")
plt.plot(x_train,regressor.predict(x_train),color= "purple")
plt.title("hours Vs scores(train)")
plt.xlabel("hours")
plt.ylabel("scores")
plt.show()
plt.scatter(x_test,y_test,color = "blue")
plt.plot(x_test,regressor.predict(x_test),color= "black")
plt.title("hours Vs scores(train)")
plt.xlabel("hours")
plt.ylabel("scores")
plt.show()
````

## Output:
![OUTPUT](output2.png)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
