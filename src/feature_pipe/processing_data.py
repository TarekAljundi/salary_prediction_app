import pandas as pd

from collections import defaultdict
from sklearn.preprocessing import LabelEncoder


decode_mapping_dict = defaultdict(list)


categorical_cols = ["experience_level", "employment_type", "company_size"]


encode_mapping_dict = {"EN": 0, "SE": 1, "MI": 2, "EX": 3, "FL": 0, "PT": 1, "FT": 2, "CT": 3, "S": 0, "M": 1, "L": 2, "X": 3}
columns_to_drop = ["Unnamed: 0","salary", "salary_currency"]
country_cols=["employee_residence", "company_location"]
remote_col = "remote_ratio"

for k, v in encode_mapping_dict.items():
    decode_mapping_dict[v].append(k)
    
def drop_unwanted_columns(df: pd.DataFrame, columns_to_drop: list[str]=columns_to_drop) -> pd.DataFrame:
    df.drop(columns=columns_to_drop, errors="ignore",inplace=True)
    return df

def encode_diff_categorical_columns(df: pd.DataFrame, categorical_cols: list[str]=categorical_cols) -> pd.DataFrame:
    
    
    for col in categorical_cols:
        df[col] = df[col].map(encode_mapping_dict)
    return df

def decode_diff_categorical_columns(df: pd.DataFrame, categorical_cols: list[str]=categorical_cols) -> pd.DataFrame:
    for col in categorical_cols:
       df[col]=df[col].map(decode_mapping_dict)
    return df

def encode_country_cols(df: pd.DataFrame) -> pd.DataFrame:
    all_locations = pd.concat([df["employee_residence"], df["company_location"]])
    encoder = LabelEncoder()
    
    encoder.fit(all_locations)
    df["employee_residence"] = encoder.transform(df["employee_residence"])
    df["company_location"] = encoder.transform(df["company_location"])
    return df, encoder

def decode_country_cols(df: pd.DataFrame, encoder: LabelEncoder, country_cols: list[str] = country_cols) -> pd.DataFrame:
    for col in country_cols:
        df[col] = df[col].map({i: encoder.inverse_transform([i])[0] for i in df[col].unique()})
    return df

def encode_remote_column(df: pd.DataFrame, remote_col: str = remote_col) -> pd.DataFrame:
    df[remote_col] = df[remote_col].replace({0: 0, 50: 1, 100: 2})
    return df

def decode_remote_column(df: pd.DataFrame, remote_col: str = remote_col) -> pd.DataFrame:
    df[remote_col] = df[remote_col].map({0: 'No Remote', 1: 'Partially Remote', 2: 'Fully Remote'})
    return df

def cast_col_to_categorical(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    df[col_name] = df[col_name].astype("category")
    return df

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_unwanted_columns(df, columns_to_drop)
    df = encode_diff_categorical_columns(df, categorical_cols)
    df, encoder = encode_country_cols(df)
    df = encode_remote_column(df, remote_col)
    return df, encoder

def decode_features(df: pd.DataFrame, encoder: LabelEncoder) -> pd.DataFrame:
    df = decode_diff_categorical_columns(df, categorical_cols)
    df = decode_country_cols(df, encoder)
    df = decode_remote_column(df, remote_col)
    return df




    



