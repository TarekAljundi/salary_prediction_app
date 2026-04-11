import pandas as pd
from pathlib import Path


from src.feature_pipe.load import load_data, split_data,split_for_train
from src.feature_pipe.processing_data import encode_features ,cast_col_to_categorical,save_encoder

from src.train_pipe.tune import study_model

from src.train_pipe.train import train_model,save_model


from src.train_pipe.predict import predict_model

cat_col = "job_title"

data = load_data()

processed_data,encoder=encode_features(data)
processed_data = cast_col_to_categorical(processed_data, cat_col)


train_df, test_df = split_data(processed_data)

X_train, y_train, X_test, y_test = split_for_train(train_df, test_df)

best_params, best_r2 = study_model(X_train, y_train, X_test, y_test,700)

r2 , model = train_model(X_train, y_train, X_test, y_test,best_params)


model_path = save_model(model)
print(f"R2 Score: {r2}")
