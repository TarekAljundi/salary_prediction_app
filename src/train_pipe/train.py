from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost
import joblib

def train_model(X_train, y_train, X_test, y_test,best_params):
    
    model = xgboost.XGBRegressor(**best_params,enable_categorical=True, random_state=42,n_jobs=-1,tree_method="hist")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = float(r2_score(y_test, y_pred))
    return r2, model

def save_model(model, file_path="model/xgb_model2.joblib"):
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")
    return file_path

def load_model(file_path="model/xgb_model2.joblib"):
    model = joblib.load(file_path)
    print(f"Model loaded from {file_path}")
    return model


