from joblib import dump
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from model_training.xgb_model import XGBModel
from model_training.svm_model import SVMModel
from model_training.random_forest_model import RFModel
from model_training.decision_tree_Model import DTModel
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import time

target = 'nivel_riesgo'

# load data
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(project_root, 'data', 'processed', 'transformed_data', 'transformed_data.csv')
dataset = pd.read_csv(file_path, delimiter=",")


# split data into X and y
X = dataset.drop(target, axis=1)
Y = dataset[target]
# split data into train and test sets
seed = 278
test_size = 0.2
val_size = 0.25
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=seed)
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
y_val = le.transform(y_val)

"""
#************************ DT ************************
"""
print("************************ DT ************************")
dt_model = DTModel(X_train, y_train, X_test, X_val, y_test, y_val)
pred, model = dt_model.run()

# save model
dump(model, "dt.dat")
print("Saved model to: dt.dat")

time.sleep( 5 )

# load model
loaded_model = load("dt.dat")
print("Loaded model from: dt.dat")


predictions = loaded_model.predict(X_val)
accuracy = accuracy_score(y_val, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
pred = None 
model = None
loaded_model = None
predictions = None
accuracy = None

"""
#************************ RF ************************
"""
print("************************ RF ************************")
rf_model = RFModel(X_train, y_train, X_test, X_val, y_test, y_val)
pred, model = rf_model.run()

# save model
dump(model, "rf.dat")
print("Saved model to: rf.dat")

time.sleep( 5 )

# load model
loaded_model = load("rf.dat")
print("Loaded model from: rf.dat")


predictions = loaded_model.predict(X_val)
accuracy = accuracy_score(y_val, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
pred = None 
model = None
loaded_model = None
predictions = None
accuracy = None

"""
#************************ SVM ************************
"""
print("************************ SVM ************************")
svm_model = SVMModel(X_train, y_train, X_test, X_val, y_test, y_val)
pred, model = svm_model.run()

# save model
dump(model, "svm.dat")
print("Saved model to: svm.dat")
 
time.sleep( 5 )

# load model
loaded_model = load("svm.dat")
print("Loaded model from: svm.dat")


predictions = loaded_model.predict(X_val)
accuracy = accuracy_score(y_val, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
pred = None 
model = None
loaded_model = None
predictions = None
accuracy = None

"""
************************ XGB ************************
"""
print("************************ XGB ************************")
# fit model on training data
xgb_model = XGBModel(X_train, y_train, X_test, X_val, y_test, y_val)
pred, model = xgb_model.run()
# save model to file
dump(model, "xgb.dat")
print("Saved model to: xgb.dat")
 
time.sleep( 5 )

# load model from file
loaded_model = load("xgb.dat")
print("Loaded model from: xgb.dat")
# make predictions for validation data
predictions = loaded_model.predict(X_val)
# evaluate predictions
accuracy = accuracy_score(y_val, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))