from numpy import loadtxt
import xgboost as xgb
from joblib import dump
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from model_training.xgb_model import XGBModel
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

target = 'Calculo_ERC'

# load data
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(project_root, 'data', 'processed', 'transformed_data', 'transformed_data.csv')
dataset = pd.read_csv(file_path, delimiter=",")


# split data into X and y
X = dataset.drop(target, axis=1)
Y = dataset[target]
# split data into train and test sets
seed = 7
test_size = 0.2
val_size = 0.25
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=seed)
# fit model on training data
xgb_model = XGBModel(X_train, y_train, X_test, X_val, y_test, y_val)
pred, model = xgb_model.run()
# save model to file
dump(model, "pima.joblib.dat")
print("Saved model to: pima.joblib.dat")
 
# load model from file
loaded_model = load("pima.joblib.dat")
print("Loaded model from: pima.joblib.dat")
# make predictions for validation data
predictions = loaded_model.predict(xgb.DMatrix(X_val))
# evaluate predictions
le = LabelEncoder()
predictions = np.argmax(predictions, axis=1)
accuracy = accuracy_score(le.fit_transform(y_val), predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))