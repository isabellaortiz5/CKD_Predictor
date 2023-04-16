import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class SVMClassifier:
    def _init_(self, file_path, target_col, test_size=0.2, random_state=42):
        self.file_path = file_path
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.svm_classifier = None
        self.grid_search = None
        self.best_params = None
        self.best_svm_classifier = None
        self.y_pred = None
        self.y_pred_optimized = None
        
    def load_data(self):
        data = pd.read_csv(self.file_path)
        self.X = data.drop(self.target_col, axis=1)
        self.y = data[self.target_col]
        
    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state)
        
    def scale_data(self):
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)
        
    def train(self, kernel='rbf', C=1, gamma='scale'):
        self.svm_classifier = SVC(kernel=kernel, C=C, gamma=gamma)
        self.svm_classifier.fit(self.X_train_scaled, self.y_train)
        
    def predict(self):
        self.y_pred = self.svm_classifier.predict(self.X_test_scaled)
        
    def optimize_params(self, param_grid, cv=5, scoring='accuracy', n_jobs=-1):
        self.grid_search = GridSearchCV(
            estimator=self.svm_classifier,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs
        )
        self.grid_search.fit(self.X_train_scaled, self.y_train)
        self.best_params = self.grid_search.best_params_
        self.best_svm_classifier = self.grid_search.best_estimator_
        
    def predict_optimized(self):
        self.y_pred_optimized = self.best_svm_classifier.predict(self.X_test_scaled)
        
    def print_confusion_matrix(self):
        print(confusion_matrix(self.y_test, self.y_pred_optimized))
        
    def print_classification_report(self):
        print(classification_report(self.y_test, self.y_pred_optimized))
        
    def print_accuracy_score(self):
        print(accuracy_score(self.y_test, self.y_pred_optimized))