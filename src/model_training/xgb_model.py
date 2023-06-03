import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import numpy as np
from joblib import dump
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

class XGBModel:
    def __init__(self, X_train, y_train, X_test, X_val, y_test, y_val):
        self.le = LabelEncoder()
        
        self.X_train = X_train

        self.y_train = self.le.fit_transform(y_train)

        self.X_test = X_test

        self.y_test = self.le.fit_transform(y_test)

        self.X_val = X_val

        self.y_val = self.le.fit_transform(y_val)

        """
        param_grid = {
            'learning_rate': [0.1, 0.01, 0.001],
            'max_depth': [3, 6, 9],
            'n_estimators': [100, 200, 300],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'subsample': [0.8, 1],
            'colsample_bytree': [0.8, 1],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [1, 1.5, 2]
        }

        """
        param_grid = {'learning_rate': [0.1, 0.01], 'max_depth': [3, 5]}
        self.param_grid = param_grid
        self.xgb_model = XGBClassifier(n_jobs=-1, early_stopping_rounds=10)

    """
    def train(self):
        self.xgb_model.fit(self.X_train, self.y_train)
    """

    def test(self):
        y_pred = self.xgb_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print("Accuracy score: {:.2f}".format(accuracy))
        return y_pred

    def tune_hyperparameters(self):
        xgb_classifier = XGBClassifier(n_jobs=-1, early_stopping_rounds=10)
        grid_search = GridSearchCV(xgb_classifier, self.param_grid, cv=5)
        grid_search.fit(self.X_train, self.y_train, eval_set=[(self.X_val, self.y_val)], verbose=False)
        self.xgb_model = grid_search.best_estimator_
        print("Best hyperparameters: {}".format(grid_search.best_params_))
        return grid_search.cv_results_
    


    def plot_confusion_matrix(self, y_pred):
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(conf_matrix, annot=True, fmt='d')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    def plot_learning_curve(self):
        results = self.xgb_model.evals_result()
        epochs = len(results['validation_0']['mlogloss'])
        x_axis = range(0, epochs)
        fig, ax = plt.subplots()
        ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
        ax.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
        ax.legend()
        plt.ylabel('Log Loss')
        plt.title('XGBoost Log Loss')
        plt.show()

    def plot_feature_importance(self):
        xgb.plot_importance(self.xgb_model)
        plt.title('Feature Importance')
        plt.show()

    def plot_hyperparameters(self, cv_results):
        cv_results_df = pd.DataFrame(cv_results)
        param_columns = [column for column in cv_results_df.columns if column.startswith('param_')]
        for column in param_columns:
            cv_results_df[column] = cv_results_df[column].astype(float)

        for param in self.param_grid.keys():
            plt.figure(figsize=(10, 6))
            sns.heatmap(cv_results_df.pivot(index='rank_test_score', columns='param_'+param, values='mean_test_score'),
                        annot=True, cmap="YlGnBu", cbar=True)
            plt.title('Hyperparameters tuning')
            plt.xlabel('Rank')
            plt.ylabel(param)
            plt.show()
    
    def train(self):
        # Set the number of threads to the number of cores for XGBoost
        params = {'n_jobs': -1}
        self.xgb_model = xgb.XGBClassifier(**params)
        self.xgb_model.fit(self.X_train, self.y_train,  eval_set=[(self.X_train, self.y_train), (self.X_val, self.y_val)], eval_metric="mlogloss", verbose=False)

        # Save the trained model to file
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(project_root, 'models')
        model_file = f'{file_path}\model.bin'
        dump(self.xgb_model, model_file)

    def run(self):
        self.train()
        y_pred = self.test()
        cv_results = self.tune_hyperparameters()
        self.train()
        y_pred = self.test()
        self.plot_confusion_matrix(y_pred)
        self.plot_learning_curve()
        self.plot_feature_importance()
        self.plot_hyperparameters(cv_results)
        return y_pred, self.xgb_model