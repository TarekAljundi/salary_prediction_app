# Salary Prediction Application



This project is an end-to-end machine learning pipeline that predicts data science salaries based on job-related features such as experience level, employment type, company size, and job title.

The system includes:
    - A trained XGBoost model (eXtreme Gradient Boosting).
    - A FastAPI prediction service
    - An LLM-powered analysis using Ollama
    - A Supabase database for storage
    - A Streamlit dashboard for visualization

## Tech Stack

    - Python
    - FastAPI
    - Streamlit
    - Supabase
    - Ollama (LLM)
    - Scikit-learn
    - Pandas
    - XGBoost
    - optuna


## Model

    - Model: XGBoost regressor
    - Target: Salary (USD)
    - Features:
    - Experience Level
    - Employment Type
    - Job Title
    - Company Size
    - Company Location
    - Employee Residence
    - Remote Ratio
    - Work Year
    - Preprocessing:
    - Categorical encoding
    - Missing value handling


  ## FastAPI Endpoint

  Root Endpoint: (To make sure it is working)
    Get /

  Prediction Endpoint: (to predict)
    POST /predict

## LLM Analysis

    A local LLM (via Ollama) is used to generate:
    - Salary explanations
    - Market insights
    - Comparative analysis
    - Suggested visualizations



## Database

    Supabase is used to store:
    - Input features
    - Predicted salary
    - LLM-generated analysis
    - Timestamp



## Dashboard Features

    - Displays predicted salaries
    - Shows LLM-generated insights
    - Visualizes salary trends with respect to job title
    - Visualizes model accuracy

## Run Locally

    1. Clone repo:
    git clone https://github.com/TarekAljundi/salary_prediction_app.git

    2. Install dependencies:
    pip install -r requirements.txt

    3. Run FastAPI:
    uvicorn src.api.api:app --reload

    4. Run Streamlit:
    streamlit run streamlit_app/app.py

    5 Run Ollama:
    ollama start (on powershell)

    6 run llm_test.py:
    python scripts\llm_test.py


## Challenges

- Handling categorical encoding consistency
- Making LLM output useful (not generic)
- Ensuring Streamlit consumes only from Supabase and going lives


## Decisions

- Used XGBoost because of internal categorical encoder
- Used optuna to finetune the model
- Used Ollama for local LLM (no API cost)
- Designed pipeline to make the app code cleaner

## Future Improvements

- Improve model accuracy
- Add user input form in Streamlit
- Go live on streamlit