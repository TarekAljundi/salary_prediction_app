from fastapi import FastAPI    
from pathlib import Path 
import pandas as pd


from src.db.save_results import save_prediction
from src.train_pipe.predict import predict_model
from src.feature_pipe.processing_data import decode_features
import joblib

from src.train_pipe.train import load_model


MODEL_PATH = Path("model/xgb_model2.joblib")



# data = pd.DataFrame({
#     "job_title": ["Data Scientist"],
#     "experience_level": ["Mid"],
#     "employment_type": ["Full-time"],
#     "remote_ratio": [100],
#     "company_size": ["Large"],
#     "salary_in_usd": [120000]
# })



app= FastAPI(title="Salary Prediction API", description="API for predicting salaries based on job data", version="1.0")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Salary Prediction API!"}

@app.post("/predict_salary")
def predict_salary(
    work_year: int,
    experience_level: str,
    employment_type: str,
    job_title: str,
    employee_residence: str,
    remote_ratio: float,
    company_location: str,
    company_size: str
    
    ):
#     # Step 1: Create DataFrame
    input_df = pd.DataFrame([{
    "work_year": work_year,
    "experience_level": experience_level,
    "employment_type": employment_type,
    "job_title": job_title,
    "employee_residence": employee_residence,
    "remote_ratio": remote_ratio,
    "company_location": company_location,
    "company_size": company_size

    }])

#     # Step 2: One-Hot Encode categorical features
  

#     # Step 3: Align with training columns
#     # for col in model_features:
#     #     if col not in input_df.columns:
#     #     input_df[col] = 0
#     #     input_df = input_df[model_features]

#     # Step 4: Predict
    
    prediction = predict_model(MODEL_PATH, input_df)
#     print(f"Predicted Salary: {prediction}")
    input_df["predicted_salary"] = prediction
    input_df=decode_features(input_df)

    for _, row in input_df.iterrows():
        save_prediction({
            "work_year": row["work_year"],
            "job_title": row["job_title"],
            "experience_level": row["experience_level"],
            "employment_type": row["employment_type"],
            "employee_residence": row["employee_residence"],
            "company_size": row["company_size"],
            "remote_ratio": row["remote_ratio"],
            "predicted_salary": row["predicted_salary"],
            "analyze": "none",
            "chart": "none"
            ,"company_location": row["company_location"]
        })

#     # Step 5: RETURN AS JSON CORRECTLY
    return {"predicted_salary": float(prediction)}

    
