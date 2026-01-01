import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
from typing import List
import time

from core.model_result import ModelResult
from core.model_definition import ModelDefinition

class ModelTrainer:
    """Handles training and evaluation of classification models."""
    
    @staticmethod
    def train_and_evaluate(X: pd.DataFrame, y: pd.Series, approaches: List[ModelDefinition]) -> List[ModelResult]:
        """Evaluates each approach using 10-fold Stratified CV, then trains on full dataset."""
        results = []
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        for approach in approaches:
            print(f"Evaluating {approach.name}...")
            start_time = time.time()
            
            # Detect column types for preprocessing
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
            
            pipeline = approach.factory_func(numeric_cols, categorical_cols)

            try:
                # Cross-validation to get predictions
                y_pred = cross_val_predict(pipeline, X, y, cv=skf, n_jobs=-1)
                
                # Calculate accuracy
                acc = accuracy_score(y, y_pred)
                n_total = len(y)
                n_correct = int(round(acc * n_total))
                
                # Train on full dataset for final model
                pipeline.fit(X, y)
                
                result = ModelResult(
                    name=approach.name,
                    pipeline=pipeline,
                    n_correct=n_correct,
                    n_total=n_total,
                    accuracy=acc
                )
                results.append(result)
                
                elapsed = time.time() - start_time
                print(f"  -> Accuracy: {acc:.4f} ({n_correct}/{n_total}) [{elapsed:.2f}s]")
                
            except Exception as e:
                print(f"  -> Failed: {e}")
                continue

        return results

    @staticmethod
    def find_best_model(results: List[ModelResult]) -> ModelResult:
        """Finds the best model based on n_correct, then accuracy."""
        if not results:
            return None
        
        sorted_results = sorted(results, key=lambda r: (r.n_correct, r.accuracy), reverse=True)
        return sorted_results[0]
