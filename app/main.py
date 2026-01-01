import sys
import os
import pandas as pd

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dataset_loader import DatasetLoader
from core.model_definition import get_all_approaches
from core.model_trainer import ModelTrainer
from core.best_model_manager import BestModelManager

def main():
    """Console-based ML classification application."""
    print("==================================================")
    print("      ML Classification Console App")
    print("==================================================")
    
    # Get CSV path from user
    csv_path = input("Enter CSV file path: ").strip()
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return

    # Load dataset
    try:
        print("Loading dataset...")
        df = DatasetLoader.load_csv(csv_path)
        X, y = DatasetLoader.split_features_target(df)
        numeric_cols, categorical_cols = DatasetLoader.detect_column_types(X)
        
        print(f"Dataset loaded: {len(df)} instances.")
        print(f"Target column: '{y.name}'")
        print(f"Features: {len(X.columns)} ({len(numeric_cols)} numeric, {len(categorical_cols)} categorical)")
    except Exception as e:
        print(f"Error loading/processing dataset: {e}")
        return

    # Train and evaluate all approaches
    print("\nStarting training and evaluation...")
    approaches = get_all_approaches()
    results = ModelTrainer.train_and_evaluate(X, y, approaches)

    if not results:
        print("No models were successfully trained.")
        return

    # Display results
    print("\n==================================================")
    print(f"{'Approach Name':<25} | {'Correct':<10} | {'Total':<8} | {'Accuracy (%)':<12}")
    print("-" * 65)
    for res in results:
        print(f"{res.name:<25} | {res.n_correct:<10} | {res.n_total:<8} | {res.accuracy * 100:.2f}%")
    print("==================================================")

    # Find and display best model
    best_model_res = ModelTrainer.find_best_model(results)
    print(f"\nBest Approach: {best_model_res.name}")
    print(f"Accuracy: {best_model_res.accuracy * 100:.2f}% ({best_model_res.n_correct}/{best_model_res.n_total})")

    manager = BestModelManager()
    manager.set_best_model(best_model_res)

    # Prediction loop
    while True:
        choice = input("\nDo you want to classify a new instance? (y/n): ").strip().lower()
        if choice != 'y':
            break
        
        print("Enter values for features:")
        feature_dict = {}
        for col in X.columns:
            val_str = input(f"  {col}: ")
            if col in numeric_cols:
                try:
                    val = float(val_str)
                except ValueError:
                    val = val_str
            else:
                val = val_str
            feature_dict[col] = val
        
        try:
            pred = manager.predict_single(feature_dict)
            print(f"Predicted Class: {pred}")
        except Exception as e:
            print(f"Prediction error: {e}")

    print("\nExiting. Goodbye!")

if __name__ == "__main__":
    main()
