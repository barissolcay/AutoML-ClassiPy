from dataclasses import dataclass
from sklearn.pipeline import Pipeline

@dataclass
class ModelResult:
    """Stores the result of a model evaluation."""
    name: str
    pipeline: Pipeline
    n_correct: int
    n_total: int
    accuracy: float
