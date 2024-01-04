import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sklearn.impute
import sklearn.compose
import sklearn.preprocessing
import sklearn.model_selection

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Salary_Data.csv')
dataset = pd.read_csv(data_path)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 1/3, random_state = 0)

# Training the Linear Regression model on the Training set
regressor = sklearn.linear_model.LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red', label="Train")
plt.plot(X_train, regressor.predict(X_train), color = 'blue', label="Regression")
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red', label="Test")
plt.plot(X_train, regressor.predict(X_train), color = 'blue', label="Regression")
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()
