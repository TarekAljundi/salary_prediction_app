import pandas as pd
from pathlib import Path


from src.feature_pipe.load import load_data, split_data
from src.feature_pipe.processing_data import encode_features , decode_features ,cast_col_to_categorical



data = load_data()

cat_col = "job_title"

print("Data Sample:")
print(data.head())

processed_data,encoder=encode_features(data)
processed_data = cast_col_to_categorical(processed_data, cat_col)
print("processed_data Sample:")
print(processed_data.info())



train_df, test_df = split_data(processed_data)
print("Train Data Sample:")
print(train_df.head())

