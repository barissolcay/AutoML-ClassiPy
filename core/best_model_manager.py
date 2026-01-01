import pandas as pd
import numpy as np
from core.model_result import ModelResult

class BestModelManager:
    """Manages the best model and handles predictions."""
    
    def __init__(self):
        self.best_result: ModelResult = None

    def set_best_model(self, result: ModelResult):
        """Sets the best model result."""
        self.best_result = result

    def predict_single(self, feature_dict: dict) -> str:
        """Predicts the class for a single instance given as a dictionary."""
        if not self.best_result:
            raise ValueError("No best model set.")

        df_new = pd.DataFrame([feature_dict])
        prediction = self.best_result.pipeline.predict(df_new)
        return prediction[0]

    def predict_batch(self, X_new: pd.DataFrame) -> np.ndarray:
        """Predicts classes for multiple instances."""
        if not self.best_result:
            raise ValueError("No best model set.")
        return self.best_result.pipeline.predict(X_new)
