from fastapi import FastAPI    
from pathlib import Path 
import pandas as pd
import joblib  



MODEL_PATH = Path("../model/xgb_model.joblib")


app= FastAPI(title="Salary Prediction API", description="API for predicting salaries based on job data", version="1.0")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Salary Prediction API!"}

@app.post("/predict")
def predict_salary(data: dict):
    # Load the model
    if not MODEL_PATH.exists():
        return {"error": f"Model not found at {str(MODEL_PATH)}"}
    model = joblib.load(MODEL_PATH)
    
    
   
    df = pd.DataFrame([data])
    if df.empty:
        return {"error": "No data provided"}
    
    pred_df = predict(df,model)
    
    resp = {"predictions": preds_df["predicted_salary"].astype(float).tolist()}
    
    return resp
    
