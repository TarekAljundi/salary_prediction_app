import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split



DATA_DIR = Path("data/ds_salaries.csv")

def load_data(data_path: Path = DATA_DIR) -> pd.DataFrame:
    
    df= pd.read_csv(data_path)

    

    
    return df

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42,output_dir: str = "data/") -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    
    
    train_df.to_csv(out_dir / "train.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    
    
    print(f"✅ Data split completed (saved to {out_dir}).")
    print(f"   Train: {train_df.shape}, Test: {test_df.shape}")
    return train_df, test_df

def split_for_train(train_df,test_df: pd.DataFrame,target_col="salary_in_usd"):
    X_train = train_df.drop(columns=[target_col])
    y_train= train_df[target_col]
    
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    return X_train, y_train, X_test, y_test


def split_for_predict(df,target_col="salary_in_usd"):

    
    X_pred = df.drop(columns=[target_col])
    y_pred = df[target_col]
    return X_pred, y_pred








