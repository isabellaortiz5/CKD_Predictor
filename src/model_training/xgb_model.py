import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

class XGBModel:
    def __init__(self, X_train, y_train, X_test, y_test, X_val, y_val, param_grid):
        le = LabelEncoder()

        self.X_train = X_train
        self.y_train = le.fit_transform(y_train)

        self.X_test = X_test
        print("************************************************Shape of y_test:", y_test.shape)
        self.y_test = le.transform(y_test)

        self.X_val = X_val
        self.y_val = le.transform(y_val)

        self.param_grid = param_grid
        self.xgb_model = XGBClassifier(n_jobs=-1, early_stopping_rounds=10)

    def test(self):
        y_pred = self.xgb_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print("Accuracy score: {:.2f}".format(accuracy))

    def tune_hyperparameters(self):
        grid_search = GridSearchCV(self.xgb_model, self.param_grid, cv=5)
        grid_search.fit(self.X_train, self.y_train)
        self.xgb_model = grid_search.best_estimator_
        print("Best hyperparameters: {}".format(grid_search.best_params_))

    def train(self):
        self.xgb_model.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)], verbose=False)

        batch_size = 1000
        n_batches = int(np.ceil(self.X_train.shape[0] / batch_size))
        batches_X = np.array_split(self.X_train, n_batches)
        batches_y = np.array_split(self.y_train, n_batches)

        self.xgb_model.n_jobs = -1
        results = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(self.xgb_model.fit)(X, y, eval_set=[(self.X_test, self.y_test)], verbose=False)
            for X, y in zip(batches_X, batches_y)
        )

        model_files = []
        for i, model in enumerate(results):
            model_file = f'model_{i}.bin'
            model.get_booster().save_model(model_file)
            model_files.append(model_file)

        self.xgb_model.load_model(model_files[0])
        for model_file in model_files[1:]:
            other_booster = xgb.Booster()
            other_booster.load_model(model_file)
            self.xgb_model.get_booster().extend(other_booster)

    def run(self):
        self.train()
        self.test()
        self.tune_hyperparameters()
        self.train()
        self.test()
        dtest = xgb.DMatrix(self.X_test, label=self.y_test)
        y_pred = self.xgb_model.predict(dtest)
        return y_pred, self.xgb_model
