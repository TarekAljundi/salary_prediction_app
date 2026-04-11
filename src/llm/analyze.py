import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ollama import chat # <- use chat function, not Ollama class
import os
from datetime import datetime

def generate_analysis(predictions: pd.DataFrame):


    os.makedirs("charts", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")




    plt.figure(figsize=(8, 5))
    sns.barplot(data=predictions, x='job_title', y='predicted_salary')
    plt.xticks(rotation=45)
    plt.title('Predicted Salary by Job Title')
    plt.tight_layout()
    chart_path = f"charts/salary_chart_{timestamp}.png"
    plt.savefig(chart_path)
    plt.close()


    text_input = predictions.to_string(index=False)

    prompt = f"""
You are a data analyst. Here is the predicted salary dataset:

{text_input}

Please generate a 200 words complete written analysis of the salary landscape.
Highlight which job titles have higher salaries, trends with remote ratio or company size,
and any other interesting insights. Refer to the chart saved as '{chart_path}'.
"""

    response = chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}],
        options={
            "num_predict": 200 
        }
    )

  
    return {
        "narrative": response.message.content,
        "chart_path": chart_path
    }