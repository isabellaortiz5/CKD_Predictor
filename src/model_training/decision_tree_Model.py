from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import numpy as np
from joblib import dump
import os

class DTModel:
    def __init__(self, X_train, y_train, X_test, X_val, y_test, y_val):
        self.le = LabelEncoder()
        
        self.X_train = X_train

        self.y_train = self.le.fit_transform(y_train)

        self.X_test = X_test

        self.y_test = self.le.fit_transform(y_test)

        self.X_val = X_val

        self.y_val = self.le.fit_transform(y_val)

        param_grid = {'criterion': ['gini', 'entropy'], 
                      'max_depth': [None, 2, 3, 5, 10],
                      'min_samples_split': [2, 5, 10]}
        self.param_grid = param_grid
        self.dt_model = DecisionTreeClassifier()

    def test(self):
        y_pred = self.dt_model.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        print("Accuracy score: {:.2f}".format(accuracy))

    def tune_hyperparameters(self):
        dt_classifier = DecisionTreeClassifier()
        grid_search = GridSearchCV(dt_classifier, self.param_grid, cv=5)
        grid_search.fit(self.X_train, self.y_train)
        self.dt_model = grid_search.best_estimator_
        print("Best hyperparameters: {}".format(grid_search.best_params_))

    def train(self):
        self.dt_model.fit(self.X_train, self.y_train)

        # Save the trained model to file
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(project_root, 'models')
        model_file = f'{file_path}\model.bin'
        dump(self.dt_model, model_file)

    def run(self):
        self.train()
        self.test()
        self.tune_hyperparameters()
        self.train()
        self.test()

        y_pred = self.dt_model.predict(self.X_test)
        return y_pred, self.dt_model
