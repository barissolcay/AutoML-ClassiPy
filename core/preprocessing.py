from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from typing import List

class PreprocessingFactory:
    """Factory class for creating preprocessing pipelines."""
    
    @staticmethod
    def get_general_transformer(numeric_cols: List[str], categorical_cols: List[str], scale_numeric: bool = True) -> ColumnTransformer:
        """
        Creates a ColumnTransformer for general algorithms.
        - Numeric: Impute (mean) -> Scale (if enabled)
        - Categorical: Impute (most_frequent) -> OneHotEncode
        """
        numeric_steps = [('imputer', SimpleImputer(strategy='mean'))]
        if scale_numeric:
            numeric_steps.append(('scaler', StandardScaler()))
        
        numeric_transformer = Pipeline(steps=numeric_steps)

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        transformers = []
        if numeric_cols:
            transformers.append(('num', numeric_transformer, numeric_cols))
        if categorical_cols:
            transformers.append(('cat', categorical_transformer, categorical_cols))

        return ColumnTransformer(transformers=transformers, remainder='drop')

    @staticmethod
    def get_discretized_transformer(numeric_cols: List[str], categorical_cols: List[str], n_bins: int = 5) -> ColumnTransformer:
        """
        Creates a ColumnTransformer for Naive Bayes.
        - Numeric: Impute (mean) -> Discretize to bins
        - Categorical: Impute (most_frequent) -> OrdinalEncode
        """
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('discretizer', KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform'))
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        
        transformers = []
        if numeric_cols:
            transformers.append(('num', numeric_transformer, numeric_cols))
        if categorical_cols:
            transformers.append(('cat', categorical_transformer, categorical_cols))

        return ColumnTransformer(transformers=transformers, remainder='drop')
