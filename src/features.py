import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Selects specific columns from a DataFrame.
    This helps keep our scikit-learn pipeline isolated from other columns.
    """
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame as input.")
        return X[self.columns].copy()

class BathroomEngineer(BaseEstimator, TransformerMixin):
    """
    Combines different bathroom metrics into a single 'TotalBath' score.
    Equation: TotalBath = FullBath + 0.5 * HalfBath + BsmtFullBath + 0.5 * BsmtHalfBath
    Fills missing values with 0 so the calculation doesn't break.
    """
    def __init__(self, full_bath="FullBath", half_bath="HalfBath", 
                 bsmt_full_bath="BsmtFullBath", bsmt_half_bath="BsmtHalfBath"):
        self.full_bath = full_bath
        self.half_bath = half_bath
        self.bsmt_full_bath = bsmt_full_bath
        self.bsmt_half_bath = bsmt_half_bath
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame as input.")
            
        X_out = X.copy()
        
        # Missing values default to 0 (meaning no bathroom of that type)
        fb = X_out[self.full_bath].fillna(0)
        hb = X_out[self.half_bath].fillna(0)
        bf = X_out[self.bsmt_full_bath].fillna(0)
        bh = X_out[self.bsmt_half_bath].fillna(0)
        
        # Calculate our combined bathroom feature
        X_out["TotalBath"] = fb + 0.5 * hb + bf + 0.5 * bh
        
        # Remove the original individual columns to avoid collinearity in the model
        raw_cols = [self.full_bath, self.half_bath, self.bsmt_full_bath, self.bsmt_half_bath]
        X_out = X_out.drop(columns=[col for col in raw_cols if col in X_out.columns])
        
        return X_out
