import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import threading
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dataset_loader import DatasetLoader
from core.model_definition import get_all_approaches
from core.model_trainer import ModelTrainer
from core.best_model_manager import BestModelManager
from core.model_result import ModelResult

class ClassificationApp:
    """Main GUI application for ML classification system."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("ML Classification System")
        self.root.geometry("1000x800")
        
        self.df = None
        self.X = None
        self.y = None
        self.numeric_cols = []
        self.categorical_cols = []
        self.best_model_manager = BestModelManager()
        
        self.filepath_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready")
        
        self.create_widgets()
        
    def create_widgets(self):
        """Creates all UI components."""
        # Dataset Section
        dataset_frame = tk.LabelFrame(self.root, text="1. Dataset Selection", padx=10, pady=10)
        dataset_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(dataset_frame, text="CSV File:").pack(side="left")
        tk.Entry(dataset_frame, textvariable=self.filepath_var, width=50).pack(side="left", padx=5)
        tk.Button(dataset_frame, text="Browse...", command=self.browse_file).pack(side="left")
        tk.Button(dataset_frame, text="Load Dataset", command=self.load_dataset).pack(side="left", padx=5)
        
        self.stats_label = tk.Label(dataset_frame, text="No dataset loaded.")
        self.stats_label.pack(side="left", padx=20)
        
        # Training Section
        train_frame = tk.LabelFrame(self.root, text="2. Algorithm Discovery", padx=10, pady=10)
        train_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        tk.Button(train_frame, text="Discover Best Algorithm", command=self.start_training).pack(anchor="w", pady=5)
        
        # Results Table
        columns = ("approach", "correct", "total", "accuracy")
        self.results_tree = ttk.Treeview(train_frame, columns=columns, show="headings", height=8)
        self.results_tree.heading("approach", text="Approach Name")
        self.results_tree.heading("correct", text="Correctly Classified")
        self.results_tree.heading("total", text="Total Instances")
        self.results_tree.heading("accuracy", text="Accuracy (%)")
        
        self.results_tree.column("approach", width=200)
        self.results_tree.column("correct", width=100)
        self.results_tree.column("total", width=100)
        self.results_tree.column("accuracy", width=100)
        
        self.results_tree.pack(fill="both", expand=True, pady=5)
        
        self.best_model_label = tk.Label(train_frame, text="Best Model: None", font=("Arial", 10, "bold"), fg="blue")
        self.best_model_label.pack(anchor="w")
        
        # Prediction Section
        self.pred_frame = tk.LabelFrame(self.root, text="3. Prediction (Best Model)", padx=10, pady=10)
        self.pred_frame.pack(fill="x", padx=10, pady=5)
        
        self.input_container = tk.Frame(self.pred_frame)
        self.input_container.pack(fill="x", expand=True)
        
        tk.Button(self.pred_frame, text="Predict Class", command=self.predict_instance).pack(pady=10)
        
        self.prediction_result_label = tk.Label(self.pred_frame, text="Prediction: -", font=("Arial", 12, "bold"))
        self.prediction_result_label.pack()
        
        # Status Bar
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor="w")
        status_bar.pack(side="bottom", fill="x")

    def browse_file(self):
        """Opens file dialog to select CSV file."""
        filename = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if filename:
            self.filepath_var.set(filename)

    def load_dataset(self):
        """Loads the selected dataset."""
        path = self.filepath_var.get()
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "Please select a valid CSV file.")
            return
            
        try:
            self.df = DatasetLoader.load_csv(path)
            self.X, self.y = DatasetLoader.split_features_target(self.df)
            self.numeric_cols, self.categorical_cols = DatasetLoader.detect_column_types(self.X)
            
            info = f"Rows: {len(self.df)} | Features: {len(self.X.columns)} (Num: {len(self.numeric_cols)}, Cat: {len(self.categorical_cols)})"
            self.stats_label.config(text=info)
            self.status_var.set("Dataset loaded successfully.")
            
            # Reset previous results
            self.results_tree.delete(*self.results_tree.get_children())
            self.best_model_label.config(text="Best Model: None")
            self.best_model_manager = BestModelManager()
            self.generate_prediction_inputs()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {e}")

    def start_training(self):
        """Starts the model training process in a separate thread."""
        if self.df is None:
            messagebox.showwarning("Warning", "Please load a dataset first.")
            return
            
        self.status_var.set("Training models... Please wait.")
        self.results_tree.delete(*self.results_tree.get_children())
        
        thread = threading.Thread(target=self.run_training_process)
        thread.start()

    def run_training_process(self):
        """Runs training and evaluation for all approaches."""
        try:
            approaches = get_all_approaches()
            results = ModelTrainer.train_and_evaluate(self.X, self.y, approaches)
            
            best_res = max(results, key=lambda r: r.accuracy)
            self.best_model_manager.set_best_model(best_res)
            
            self.root.after(0, self.update_results_ui, results, best_res)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Training failed: {e}"))
            self.root.after(0, lambda: self.status_var.set("Training failed."))

    def update_results_ui(self, results, best_res):
        """Updates the results table in the UI."""
        self.results_tree.tag_configure('best', background='#d1e7dd')

        for res in results:
            tags = ('best',) if res == best_res else ()
            self.results_tree.insert(
                "", 
                "end", 
                values=(res.name, res.n_correct, res.n_total, f"{res.accuracy:.2%}"),
                tags=tags
            )
            
        self.best_model_label.config(text=f"Best Model: {best_res.name} ({best_res.accuracy:.2%})")
        self.status_var.set("Training complete.")
        messagebox.showinfo("Success", f"Training complete. Best model: {best_res.name}")

    def generate_prediction_inputs(self):
        """Creates input fields for making predictions."""
        for widget in self.input_container.winfo_children():
            widget.destroy()
            
        self.input_vars = {}
        row = 0
        col = 0
        
        # Numeric inputs (textbox)
        for feature in self.numeric_cols:
            frame = tk.Frame(self.input_container)
            frame.grid(row=row, column=col, padx=10, pady=5, sticky="w")
            
            tk.Label(frame, text=f"{feature} (Numeric):").pack(anchor="w")
            var = tk.DoubleVar()
            entry = tk.Entry(frame, textvariable=var)
            entry.pack(fill="x")
            self.input_vars[feature] = var
            
            col += 1
            if col > 3:
                col = 0
                row += 1
                
        # Categorical inputs (dropdown)
        for feature in self.categorical_cols:
            frame = tk.Frame(self.input_container)
            frame.grid(row=row, column=col, padx=10, pady=5, sticky="w")
            
            tk.Label(frame, text=f"{feature} (Nominal):").pack(anchor="w")
            
            unique_vals = sorted(self.X[feature].unique().tolist())
            var = tk.StringVar()
            combo = ttk.Combobox(frame, textvariable=var, values=unique_vals, state="readonly")
            combo.pack(fill="x")
            if unique_vals:
                combo.current(0)
            self.input_vars[feature] = var
            
            col += 1
            if col > 3:
                col = 0
                row += 1

    def predict_instance(self):
        """Makes a prediction using the best model."""
        if self.best_model_manager.best_result is None:
            messagebox.showwarning("Warning", "Please train models first.")
            return
            
        try:
            input_data = {}
            for feature, var in self.input_vars.items():
                val = var.get()
                input_data[feature] = val
                
            prediction = self.best_model_manager.predict_single(input_data)
            self.prediction_result_label.config(text=f"Prediction: {prediction}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ClassificationApp(root)
    root.mainloop()
