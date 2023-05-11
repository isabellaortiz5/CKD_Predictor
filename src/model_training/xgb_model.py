import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder



"""
class XGBModel:
    def __init__(self, X_train, y_train, X_test, y_test, param_grid):
        self.params = {
            'use_label_encoder': False,
            'eval_metric': 'mlogloss',
            'objective': 'multi:softmax',
        }
        self.grid_search = None
        self.best_model = None
        self.X_train = X_train
        self.y_train = self._prepare_target(y_train)
        self.X_test = X_test
        self.y_test = self._prepare_target(y_test)
        self.param_grid = param_grid
        self.y_pred = None
        self.params['num_class'] = len(set(self.y_train))

    def _prepare_target(self, y):
        label_encoder = LabelEncoder()
        return label_encoder.fit_transform(y)

    def hyperparameter_tuning(self, scoring='accuracy', cv=3, verbose=1):
        model = xgb.XGBClassifier(**self.params)
        self.grid_search = GridSearchCV(model, self.param_grid, scoring=scoring, cv=cv, verbose=verbose)
        self.grid_search.fit(self.X_train, self.y_train)
        self.params.update(self.grid_search.best_params_)
        print("Best parameters: ", self.params)

    def train(self):
        self.best_model = xgb.XGBClassifier(**self.params)
        self.best_model.fit(self.X_train, self.y_train)

    def evaluate(self):
        self.y_pred = self.best_model.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, self.y_pred)
        print("Accuracy: %.2f%%" % (accuracy * 100))

        print("Classification Report:\n", classification_report(self.y_test, self.y_pred))

    def run(self):
        self.hyperparameter_tuning()
        self.train()
        self.evaluate()


        class XGBoostModel:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.model = xgb.XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    # Train and evaluate model
    model = XGBoostModel(n_estimators=200, learning_rate=0.05, max_depth=5)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Calculate accuracy of the model
    accuracy = np.sum(predictions == y_test) / len(y_test)
    print("Accuracy:", accuracy)


    import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score



"""

"""
class XGBModel:
    def __init__(self, X_train, y_train, param_grid):
        self.X_train = X_train
        self.y_train = y_train
        self.param_grid = param_grid

    def run(self):
        xgb_model = xgb.XGBClassifier()
        grid_search = GridSearchCV(xgb_model, self.param_grid, cv=5)
        grid_search.fit(self.X_train, self.y_train)
        best_params = grid_search.best_params_

        # Training the model with the best hyperparameters
        xgb_model = xgb.XGBClassifier(**best_params)
        xgb_model.fit(self.X_train, self.y_train)

        # Predicting on the test set
        y_pred = xgb_model.predict(self.X_test)

        # Printing the accuracy of the model
        accuracy = (y_pred == self.y_test).mean()
        print("Accuracy:", accuracy)
"""

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib
import os

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
        # Convert the numpy array to a DMatrix object
        dtest = xgb.DMatrix(self.X_test)
        
        y_pred_proba = self.xgb_model.predict(dtest)
        y_pred = np.argmax(y_pred_proba, axis=1)

        accuracy = accuracy_score(self.y_test, y_pred)
        print("Accuracy score: {:.2f}".format(accuracy))

    def tune_hyperparameters(self):
        xgb_classifier = XGBClassifier(n_jobs=-1, early_stopping_rounds=10)
        grid_search = GridSearchCV(xgb_classifier, self.param_grid, cv=5)
        grid_search.fit(self.X_train, self.y_train, eval_set=[(self.X_val, self.y_val)], verbose=False)
        self.xgb_model = grid_search.best_estimator_
        print("Best hyperparameters: {}".format(grid_search.best_params_))

    def train(self):
        self.xgb_model.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)], verbose=False)

        # Split the training data into batches
        batch_size = 1000
        n_batches = int(np.ceil(self.X_train.shape[0] / batch_size))
        batches_X = np.array_split(self.X_train, n_batches)
        batches_y = np.array_split(self.y_train, n_batches)

        # Parallelize the training of each batch using joblib
        self.xgb_model.n_jobs = -1  # Use all available CPU cores
        results = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(self.xgb_model.fit)(X, y, eval_set=[(self.X_test, self.y_test)], verbose=False)
            for X, y in zip(batches_X, batches_y)
        )

        # Save the trained models to files
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(project_root, 'models')
        model_files = []
        for i, model in enumerate(results):
            
            model_file = f'{file_path}\model_{i}.bin'
            model.get_booster().save_model(model_file)
            model_files.append(model_file)

        # Load the first model to initialize the xgb.Booster object
        self.xgb_model = xgb.Booster(model_file=model_files[0])

        # Merge the results from each batch by loading and combining their boosters
        for model_file in model_files[1:]:
            other_booster = xgb.Booster(model_file=model_file)
            self.xgb_model = self.xgb_model + other_booster    

    def run(self):
        self.train()
        self.test()
        self.tune_hyperparameters()
        self.train()
        self.test()

        y_pred = self.xgb_model.predict(xgb.DMatrix(self.X_test))
        return y_pred, self.xgb_model