


import joblib

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost
import optuna


def study_model(X_train, y_train, X_test, y_test,n_trials=15):


    def objective(trial):
        params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
        'enable_categorical':True
        }
        

        model = xgboost.XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        r2 = float(r2_score(y_test, y_pred))
        return r2
        
    study = optuna.create_study(direction="maximize")
    study.optimize(objective,n_trials)

    print("Best trial params: ", study.best_trial.params)
    print("Best trial r2: ", study.best_value)
    
    return study.best_trial.params, study.best_value





