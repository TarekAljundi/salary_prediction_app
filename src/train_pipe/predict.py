import xgboost as xgb

from src.feature_pipe.processing_data import load_encoder,encode_features,decode_features,cast_col_to_categorical
from src.feature_pipe.load import split_for_predict
import joblib

cat_col = "job_title"


def predict_model(model, df):
    processed_df, _ = encode_features(df)
    
    processed_df = cast_col_to_categorical(processed_df, cat_col)
   
    
    
    model = joblib.load(model)
   
    y_pred = model.predict(processed_df)
    
   
    
    
    
    # X_pred = decode_features(X_pred, encoder)
    # X_pred["predicted_salary"] = y_pred
    
    
    return y_pred[0]



