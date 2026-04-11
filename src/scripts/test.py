import pandas as pd
from pathlib import Path


from src.feature_pipe.load import load_data, split_data,split_for_train
from src.feature_pipe.processing_data import encode_features ,cast_col_to_categorical,save_encoder

from src.train_pipe.tune import study_model

from src.train_pipe.train import train_model,save_model


from src.train_pipe.predict import predict_model





data ={"work_year": 2022, "experience_level": "SE", "employment_type": "FT", "job_title": "Data Scientist", "employee_residence": "US","remote_ratio": 0,'company_location': "US", "company_size": "S"}

data2={    "work_year": 2022,
    "experience_level": "SE",
    "employment_type": "FT",
    "job_title": "Data Scientist",
    "employee_residence": "US",
    "remote_ratio": 0,
    "company_location": "US",
    "company_size": "S"}




test_df= pd.read_csv("data/test.csv")

df = pd.DataFrame([data2])

model_path = Path("model/xgb_model2.joblib")

print(predict_model(model_path,df))












