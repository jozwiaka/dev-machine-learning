import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sklearn.impute
import sklearn.compose
import sklearn.preprocessing
import sklearn.model_selection

#os.system('cls' if os.name == 'nt' else 'clear')

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Data.csv')
dataset = pd.read_csv(data_path)
X = dataset.iloc[:, :-1].values # 0:n-1 columns
y = dataset.iloc[:, -1].values
print(f"X = {X}")
print(f"y = {y}\n")

# Estimate missing data
imputer = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:,1:3] = imputer.transform(X[:,1:3])
print(f"X = {X}\n")

# Encode the independent variable
ct = sklearn.compose.ColumnTransformer(transformers=[('encoder', sklearn.preprocessing.OneHotEncoder(), [0])], remainder='passthrough')
X=np.array(ct.fit_transform(X))
print(f"X = {X}\n")

# Encode the dependent variable
le = sklearn.preprocessing.LabelEncoder()
y = le.fit_transform(y)
print(f"y = {y}\n")

# Train test split
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=1)
# test_size=0.2 -> split data (X, y):
# X_train = 0.8*X
# x_test = 0.2*X
 
# random_state=1 -> data split is the same each run
# #remove random_state -> the data split will vary between runs
print(f"X_train = {X_train}")
print(f"X_test = {X_test}")
print(f"y_train = {y_train}")
print(f"y_test = {y_test}\n")

# Scale feature
sc = sklearn.preprocessing.StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print(f"X_train = {X_train}")
print(f"X_test = {X_test}\n")
