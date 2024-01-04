import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sklearn.compose
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.linear_model

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "50_Startups.csv")
dataset = pd.read_csv(data_path)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)

ct = sklearn.compose.ColumnTransformer(
    transformers=[("encoder", sklearn.preprocessing.OneHotEncoder(), [3])],
    remainder="passthrough",
)
X = np.array(ct.fit_transform(X))
print(X)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, test_size=0.2, random_state=0
)

regressor = sklearn.linear_model.LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(
    np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)
)
