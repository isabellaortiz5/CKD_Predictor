from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import dump
import os

class RFModel:
    def __init__(self, X_train, y_train, X_test, X_val, y_test, y_val):
        self.le = LabelEncoder()
        
        self.X_train = X_train
        self.y_train = self.le.fit_transform(y_train)
        self.X_test = X_test
        self.y_test = self.le.fit_transform(y_test)
        self.X_val = X_val
        self.y_val = self.le.fit_transform(y_val)
        """
        rf_params = {
            'n_estimators': [50, 100, 200],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
        """
        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]}
        self.param_grid = param_grid
        self.rf_model = RandomForestClassifier(n_jobs=-1)

    def test(self):
        y_pred = self.rf_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print("Accuracy score: {:.2f}".format(accuracy))
        return y_pred

    def tune_hyperparameters(self):
        rf_classifier = RandomForestClassifier(n_jobs=-1)
        grid_search = GridSearchCV(rf_classifier, self.param_grid, cv=5)
        grid_search.fit(self.X_train, self.y_train)
        self.rf_model = grid_search.best_estimator_
        print("Best hyperparameters: {}".format(grid_search.best_params_))

    def train(self):
        self.rf_model.fit(self.X_train, self.y_train)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(project_root, 'models')
        model_file = f'{file_path}\model.bin'
        dump(self.rf_model, model_file)

    def plot_confusion_matrix(self, y_pred):
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(conf_matrix, annot=True, fmt='d')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('rfConfusion.png')

    def plot_learning_curve(self):
        train_sizes, train_scores, test_scores = learning_curve(self.rf_model, self.X_train, self.y_train, cv=5)
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)

        plt.figure()
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.legend(loc="best")
        plt.grid()
        plt.savefig('rfLearning.png')
        plt.clf()

    def plot_feature_importance(self):
        plt.figure()
        plt.rcParams['figure.figsize'] = [15, 25]
        plt.bar(self.X_train.columns, self.rf_model.feature_importances_)
        plt.title('Feature Importance')
        plt.xticks(rotation='vertical')
        plt.savefig('rfFeature.png')
        

    def run(self):
        self.train()
        y_pred = self.test()
        self.tune_hyperparameters()
        self.train()
        y_pred = self.test()
        self.plot_confusion_matrix(y_pred)
        self.plot_learning_curve()
        self.plot_feature_importance()
        return y_pred, self.rf_model
