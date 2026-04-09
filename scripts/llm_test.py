import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train_pipe.predict import predict_model

from src.llm.analyze import generate_analysis

from src.db.save_results import save_prediction

MODEL_PATH = "model/xgb_model2.joblib"

# Example: 3 jobs
jobs = [
    {"work_year": 2022,
    "experience_level": "SE",
    "employment_type": "FT",
    "job_title": "Data Scientist",
    "employee_residence": "US",
    "remote_ratio": 0,
    "company_location": "US",
    "company_size": "S"},
    {"work_year": 2023,
    "experience_level": "MI",
    "employment_type": "PT",
    "job_title": "Research Scientist",
    "employee_residence": "DE",
    "remote_ratio": 50,
    "company_location": "DE",
    "company_size": "L"},
        {"work_year": 2024,
    "experience_level": "EN",
    "employment_type": "FL",
    "job_title": "Data Analyst",
    "employee_residence": "IN",
    "remote_ratio": 50,
    "company_location": "IN",
    "company_size": "M"},
    {"work_year": 2024,
    "experience_level": "EN",
    "employment_type": "FL",
    "job_title": "Data Engineer",
    "employee_residence": "IN",
    "remote_ratio": 50,
    "company_location": "IN",
    "company_size": "M"},
       {"work_year": 2024,
    "experience_level": "EN",
    "employment_type": "FL",
    "job_title": "Machine Learning Engineer",
    "employee_residence": "IN",
    "remote_ratio": 50,
    "company_location": "IN",
    "company_size": "M"},
    {"work_year": 2022,
    "experience_level": "EN",
    "employment_type": "FT",
    "job_title": "Business Data Analyst",
    "employee_residence": "DE",
    "remote_ratio": 50,
    "company_location": "IN",
    "company_size": "M"},
    {"work_year": 2025,
    "experience_level": "SE",
    "employment_type": "FT",
    "job_title": "Lead Data Scientist",
    "employee_residence": "IN",
    "remote_ratio": 50,
    "company_location": "US",
    "company_size": "L"}

]

# Call API for each job and collect results
data = []
for job in jobs:
    salary = predict_model(MODEL_PATH, pd.DataFrame([job]))
    job['predicted_salary'] = salary
    data.append(job)

df = pd.DataFrame(data)

# Generate LLM analysis
result = generate_analysis(df)

for _, row in df.iterrows():
    save_prediction({
        "work_year": row["work_year"],
        "job_title": row["job_title"],
        "experience_level": row["experience_level"],
        "employment_type": row["employment_type"],
        "employee_residence": row["employee_residence"],
        "company_size": row["company_size"],
        "remote_ratio": row["remote_ratio"],
        "predicted_salary": row["predicted_salary"],
        "analyze": result["narrative"],
        "chart": result["chart_path"]
        ,"company_location": row["company_location"]
    })

print(result['narrative'])       # narrative text from LLM
print("Chart saved at:", result['chart_path'])