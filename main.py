import pandas as pd
from pathlib import Path


from src.feature_pipe.load import load_data, split_data
from src.feature_pipe.processing_data import encode_features , decode_features ,cast_col_to_categorical

cat_col = "job_title"

data = load_data()



processed_data,encoder=encode_features(data)
processed_data = cast_col_to_categorical(processed_data, cat_col)

processed_data.head()


train_df, test_df = split_data(processed_data)





