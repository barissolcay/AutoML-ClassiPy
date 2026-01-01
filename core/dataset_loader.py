import pandas as pd
import numpy as np
from typing import Tuple, List

class DatasetLoader:
    """Handles loading and processing CSV datasets."""
    
    @staticmethod
    def load_csv(filepath: str) -> pd.DataFrame:
        """Loads a CSV file into a DataFrame."""
        try:
            df = pd.read_csv(filepath)
            return df
        except Exception as e:
            raise IOError(f"Error loading CSV file: {e}")

    @staticmethod
    def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Splits DataFrame into features (X) and target (y). Last column is target."""
        if df.shape[1] < 2:
            raise ValueError("Dataset must have at least one feature and one target column.")
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        return X, y

    @staticmethod
    def detect_column_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Detects numeric and categorical columns in the feature set."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        return numeric_cols, categorical_cols
