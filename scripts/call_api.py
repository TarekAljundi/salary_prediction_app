# import requests

# API_URL = "http://127.0.0.1:8000/predict_salary"

# def get_predict_salary(
#     work_year: int,
#     experience_level: str,
#     employment_type: str,
#     job_title: str,
#     employee_residence: str,
#     remote_ratio: float,
#     company_location: str,
#     company_size: str
    
#     ):

#     try:
#         response = requests.get(API_URL, params=params)
#         response.raise_for_status()
#         data = response.json()
#         if "predicted_salary" in data:
#             return data["predicted_salary"]
#         else:
#             return f"Error from API: {data}"
#     except requests.exceptions.RequestException as e:
#         return f"Request failed: {e}"

# # --- MAIN ---
# if __name__ == "__main__":
#     salary = get_predict_salary(
#        work_year= 2022,
#         experience_level= "SE",
#         employment_type= "FT",
#         job_title= "Data Scientist",
#         employee_residence= "US",
#         remote_ratio= 0,
#         company_location= "US",
#         company_size= "S"
#     )

#     if isinstance(salary, float):
#         print(f"Predicted Salary: ${salary:.2f}")
#     else:
#         print(salary)