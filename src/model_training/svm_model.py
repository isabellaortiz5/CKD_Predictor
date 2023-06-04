from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import dump
import os

class SVMModel:
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
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [2, 3, 4, 5],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1],
            'coef0': [0, 1, 2, 3]
        }

        """
        param_grid = {'gamma': ['scale', 'auto']}
        self.param_grid = param_grid
        self.svm_model = SVC()

    def test(self):
        y_pred = self.svm_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print("Accuracy score: {:.2f}".format(accuracy))
        return y_pred

    def tune_hyperparameters(self):
        svm_classifier = SVC()
        grid_search = GridSearchCV(svm_classifier, self.param_grid, cv=2)
        grid_search.fit(self.X_train, self.y_train)
        self.svm_model = grid_search.best_estimator_
        print("Best hyperparameters: {}".format(grid_search.best_params_))
        return grid_search.cv_results_

    def train(self):
        self.svm_model.fit(self.X_train, self.y_train)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(project_root, 'models')
        model_file = f'{file_path}\model.bin'
        dump(self.svm_model, model_file)

    def plot_confusion_matrix(self, y_pred):
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(conf_matrix, annot=True, fmt='d')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('svmConfusion.png')
        plt.clf()

    def plot_learning_curve(self):
        train_sizes, train_scores, test_scores = learning_curve(self.svm_model, self.X_train, self.y_train, cv=5)
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)

        plt.figure()
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.legend(loc="best")
        plt.grid()
        plt.savefig('svmLearning.png')
        plt.clf()

    def run(self):
        self.train()
        y_pred = self.test()
        self.tune_hyperparameters()
        self.train()
        y_pred = self.test()
        self.plot_confusion_matrix(y_pred)
        self.plot_learning_curve()
        return y_pred, self.svm_model
