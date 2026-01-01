import sys
import os
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dataset_loader import DatasetLoader
from core.model_definition import get_all_approaches
from core.model_trainer import ModelTrainer

def run_tests():
    """Automated test runner for the classification system."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    datasets = ["heart.csv"]

    print("==================================================")
    print("      Automated Test Runner")
    print("==================================================")

    for filename in datasets:
        filepath = os.path.join(project_root, filename)
        print(f"\nTesting Dataset: {filename}")
        
        if not os.path.exists(filepath):
            print(f"  [Skipped] File not found: {filepath}")
            continue

        try:
            # Load data
            df = DatasetLoader.load_csv(filepath)
            
            # Limit large datasets for faster testing
            MAX_ROWS = 10000
            if len(df) > MAX_ROWS:
                print(f"  Dataset too large ({len(df)} rows). Sampling first {MAX_ROWS} rows.")
                df = df.iloc[:MAX_ROWS]
            
            X, y = DatasetLoader.split_features_target(df)
            
            # Check for valid target
            if y.nunique() < 2:
                print("  [Skipped] Target has less than 2 classes.")
                continue

            print(f"  Loaded {len(df)} rows. Target: {y.name}")

            # Train and evaluate
            approaches = get_all_approaches()
            results = ModelTrainer.train_and_evaluate(X, y, approaches)

            if not results:
                print("  No results obtained.")
                continue

            best = ModelTrainer.find_best_model(results)
            print(f"  BEST MODEL: {best.name} (Acc: {best.accuracy:.4f})")
            
        except Exception as e:
            print(f"  [Error] Failed to process {filename}: {e}")

    print("\nTest Runner Completed.")

if __name__ == "__main__":
    run_tests()
