import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

data = pd.read_csv(r"C:\Users\HONOR\Desktop\ML_projects\Linear_regression\student-mat.csv",sep=';')

data = data[['G1','G2','G3','studytime','failures','schoolsup','activities','internet']]

data = pd.get_dummies(data, columns=['schoolsup', 'activities', 'internet'], drop_first=True)

predict = 'G3'

X = np.array(data.drop([predict],axis = 1))
y = np.array(data[predict])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

linear = LinearRegression()

linear.fit(X_train, y_train)
accuracy = linear.score(X_test, y_test)
print(accuracy)

print("coefficients:",linear.coef_)
print('Intercept:',linear.intercept_)

y_pred = linear.predict(X_test)

for x in range(len(y_pred)):
    print(y_pred[x],X_test[x],y_test[x])
