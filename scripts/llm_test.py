import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train_pipe.predict import predict_model

from src.llm.analyze import generate_analysis

from src.db.save_results import save_prediction

MODEL_PATH = "model/xgb_model2.joblib"



# _____
jobs = [
{"work_year": 2023, "experience_level": "SE", "employment_type": "FT", "job_title": "Big Data Architect", "employee_residence": "US", "remote_ratio": 100, "company_location": "US", "company_size": "L"},
{"work_year": 2021, "experience_level": "EN", "employment_type": "PT", "job_title": "Data Analytics Engineer", "employee_residence": "FR", "remote_ratio": 0, "company_location": "FR", "company_size": "S"},
{"work_year": 2024, "experience_level": "MI", "employment_type": "FT", "job_title": "Applied Machine Learning Scientist", "employee_residence": "DE", "remote_ratio": 50, "company_location": "DE", "company_size": "M"},
{"work_year": 2022, "experience_level": "SE", "employment_type": "CT", "job_title": "Cloud Data Engineer", "employee_residence": "NL", "remote_ratio": 100, "company_location": "NL", "company_size": "L"},
{"work_year": 2020, "experience_level": "EN", "employment_type": "FL", "job_title": "Data Analyst", "employee_residence": "IN", "remote_ratio": 0, "company_location": "IN", "company_size": "S"},
{"work_year": 2025, "experience_level": "EX", "employment_type": "FT", "job_title": "Director of Data Science", "employee_residence": "GB", "remote_ratio": 100, "company_location": "GB", "company_size": "L"},
{"work_year": 2023, "experience_level": "MI", "employment_type": "FT", "job_title": "Machine Learning Developer", "employee_residence": "CA", "remote_ratio": 50, "company_location": "CA", "company_size": "M"},
{"work_year": 2021, "experience_level": "EN", "employment_type": "PT", "job_title": "BI Data Analyst", "employee_residence": "ES", "remote_ratio": 0, "company_location": "ES", "company_size": "S"},
{"work_year": 2024, "experience_level": "SE", "employment_type": "FT", "job_title": "Principal Data Scientist", "employee_residence": "CH", "remote_ratio": 100, "company_location": "CH", "company_size": "L"},
{"work_year": 2022, "experience_level": "MI", "employment_type": "CT", "job_title": "ETL Developer", "employee_residence": "PL", "remote_ratio": 50, "company_location": "PL", "company_size": "M"},

{"work_year": 2020, "experience_level": "EN", "employment_type": "FL", "job_title": "Product Data Analyst", "employee_residence": "MX", "remote_ratio": 0, "company_location": "MX", "company_size": "S"},
{"work_year": 2025, "experience_level": "EX", "employment_type": "FT", "job_title": "Head of Machine Learning", "employee_residence": "AE", "remote_ratio": 100, "company_location": "AE", "company_size": "L"},
{"work_year": 2023, "experience_level": "MI", "employment_type": "FT", "job_title": "AI Scientist", "employee_residence": "SG", "remote_ratio": 50, "company_location": "SG", "company_size": "M"},
{"work_year": 2021, "experience_level": "EN", "employment_type": "PT", "job_title": "Finance Data Analyst", "employee_residence": "IT", "remote_ratio": 0, "company_location": "IT", "company_size": "S"},
{"work_year": 2024, "experience_level": "SE", "employment_type": "FT", "job_title": "Lead Data Engineer", "employee_residence": "AU", "remote_ratio": 100, "company_location": "AU", "company_size": "L"},
{"work_year": 2022, "experience_level": "MI", "employment_type": "CT", "job_title": "Data Science Engineer", "employee_residence": "TR", "remote_ratio": 50, "company_location": "TR", "company_size": "M"},
{"work_year": 2020, "experience_level": "EN", "employment_type": "FL", "job_title": "Business Data Analyst", "employee_residence": "RO", "remote_ratio": 0, "company_location": "RO", "company_size": "S"},
{"work_year": 2025, "experience_level": "EX", "employment_type": "FT", "job_title": "Head of Data Science", "employee_residence": "CA", "remote_ratio": 100, "company_location": "CA", "company_size": "L"},
{"work_year": 2023, "experience_level": "MI", "employment_type": "FT", "job_title": "Computer Vision Engineer", "employee_residence": "JP", "remote_ratio": 50, "company_location": "JP", "company_size": "M"},
{"work_year": 2021, "experience_level": "EN", "employment_type": "PT", "job_title": "Marketing Data Analyst", "employee_residence": "BR", "remote_ratio": 0, "company_location": "BR", "company_size": "S"}
]


data = []
for job in jobs:
    salary = predict_model(MODEL_PATH, pd.DataFrame([job]))
    job['predicted_salary'] = salary
    data.append(job)

df = pd.DataFrame(data)


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

print(result['narrative']) 
print("Chart saved at:", result['chart_path'])