from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import CategoricalNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from typing import List, Callable

from core.preprocessing import PreprocessingFactory

class ModelDefinition:
    """Holds the name and factory function for a classification approach."""
    def __init__(self, name: str, factory_func: Callable[[List[str], List[str]], Pipeline]):
        self.name = name
        self.factory_func = factory_func

def get_all_approaches() -> List[ModelDefinition]:
    """Returns a list of all classification approaches to be evaluated."""
    approaches = []

    # 1. Naive Bayes with discretized features
    def nb_factory(num, cat):
        preprocessor = PreprocessingFactory.get_discretized_transformer(num, cat, n_bins=5)
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', CategoricalNB())
        ])
    approaches.append(ModelDefinition("NaiveBayes_discretized", nb_factory))

    # 2. Logistic Regression with scaling
    def logreg_factory(num, cat):
        preprocessor = PreprocessingFactory.get_general_transformer(num, cat, scale_numeric=True)
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, solver='lbfgs'))
        ])
    approaches.append(ModelDefinition("Logistic_scaled", logreg_factory))

    # 3. KNN with k=1
    def knn1_factory(num, cat):
        preprocessor = PreprocessingFactory.get_general_transformer(num, cat, scale_numeric=True)
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', KNeighborsClassifier(n_neighbors=1, algorithm='auto'))
        ])
    approaches.append(ModelDefinition("KNN_k1_scaled", knn1_factory))

    # 4. KNN with k=3
    def knn3_factory(num, cat):
        preprocessor = PreprocessingFactory.get_general_transformer(num, cat, scale_numeric=True)
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', KNeighborsClassifier(n_neighbors=3, algorithm='auto'))
        ])
    approaches.append(ModelDefinition("KNN_k3_scaled", knn3_factory))

    # 5. KNN with k=5
    def knn5_factory(num, cat):
        preprocessor = PreprocessingFactory.get_general_transformer(num, cat, scale_numeric=True)
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', KNeighborsClassifier(n_neighbors=5, algorithm='auto'))
        ])
    approaches.append(ModelDefinition("KNN_k5_scaled", knn5_factory))

    # 6. Decision Tree (J48 equivalent)
    def dt_factory(num, cat):
        preprocessor = PreprocessingFactory.get_general_transformer(num, cat, scale_numeric=False)
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier(random_state=42))
        ])
    approaches.append(ModelDefinition("DecisionTree_basic", dt_factory))

    # 7. Random Forest
    def rf_factory(num, cat):
        preprocessor = PreprocessingFactory.get_general_transformer(num, cat, scale_numeric=False)
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
        ])
    approaches.append(ModelDefinition("RandomForest_basic", rf_factory))

    # 8. Extra Trees (Random Trees equivalent)
    def et_factory(num, cat):
        preprocessor = PreprocessingFactory.get_general_transformer(num, cat, scale_numeric=False)
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1))
        ])
    approaches.append(ModelDefinition("ExtraTrees_basic", et_factory))

    # 9. Multilayer Perceptron (ANN)
    def mlp_factory(num, cat):
        preprocessor = PreprocessingFactory.get_general_transformer(num, cat, scale_numeric=True)
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42, early_stopping=True))
        ])
    approaches.append(ModelDefinition("MLP_scaled", mlp_factory))

    # 10. SVM with linear kernel
    def svm_lin_factory(num, cat):
        preprocessor = PreprocessingFactory.get_general_transformer(num, cat, scale_numeric=True)
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LinearSVC(random_state=42, max_iter=2000, dual='auto'))
        ])
    approaches.append(ModelDefinition("SVM_linear_scaled", svm_lin_factory))

    # 11. SVM with RBF kernel
    def svm_rbf_factory(num, cat):
        preprocessor = PreprocessingFactory.get_general_transformer(num, cat, scale_numeric=True)
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', SVC(kernel='rbf', random_state=42, cache_size=500))
        ])
    approaches.append(ModelDefinition("SVM_rbf_scaled", svm_rbf_factory))

    return approaches
