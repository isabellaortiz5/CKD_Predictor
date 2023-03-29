import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report


class XGBModel:
    def __init__(self, X_train, y_train, X_test, y_test, param_grid ):
        self.params = params or {
            'use_label_encoder': False,
            'eval_metric': 'mlogloss',
            'objective': 'multi:softmax',
            'num_class': 5,
        }
        self.grid_search = None
        self.best_model = None
        self.X_train = X_train
        self.y_train = y_train 
        self.X_test = X_test 
        self.y_test = y_test
        self.param_grid = param_grid
        self.y_pred = None
 
    def hyperparameter_tuning(self, scoring='accuracy', cv=5, verbose=1):
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
        self.hyperparameter_tuning(self.X_train, self.y_train, self.param_grid)
        self.train(self.X_train, self.y_train)
        self.evaluate(self.X_test, self.y_test)