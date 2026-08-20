import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold

from src.config import RANDOM_STATE, ALPHA_GRID, L1_RATIOS, CROSS_VALIDATION_FOLDS
from src.features import ColumnSelector, BathroomEngineer

def build_preprocessing_pipeline():
    """
    Creates our data preprocessing workflow.
    Selects columns, combines bathrooms, handles missing entries, and scales the inputs.
    """
    cols_to_use = ["GrLivArea", "BedroomAbvGr", "FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]
    
    return Pipeline([
        ("selector", ColumnSelector(columns=cols_to_use)),
        ("bath_engineer", BathroomEngineer(
            full_bath="FullBath", 
            half_bath="HalfBath", 
            bsmt_full_bath="BsmtFullBath", 
            bsmt_half_bath="BsmtHalfBath"
        )),
        ("imputer", SimpleImputer(strategy="median")),  # Fills missing values with the column median
        ("scaler", StandardScaler())                     # Standardizes features to have mean=0 and variance=1
    ])

def calculate_rmsle(y_true, y_pred):
    """
    Computes Root Mean Squared Logarithmic Error (RMSLE).
    Clips predictions to 0 to prevent issues with log(negative number).
    """
    y_true_clipped = np.clip(y_true, 0, None)
    y_pred_clipped = np.clip(y_pred, 0, None)
    return np.sqrt(mean_squared_error(np.log1p(y_true_clipped), np.log1p(y_pred_clipped)))

def evaluate_predictions(y_true, y_pred):
    """
    Calculates regression metrics (RMSE, MAE, R2, and RMSLE) to evaluate a model's performance.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmsle = calculate_rmsle(y_true, y_pred)
    
    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "RMSLE": rmsle
    }

def run_cross_validation(X, y, pipeline, model, use_log_target=True):
    """
    Splits the data into folds, preprocesses, trains, and evaluates a model.
    If use_log_target is True, trains on log1p(y) and back-transforms using expm1.
    """
    kf = KFold(n_splits=CROSS_VALIDATION_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    metrics_list = []
    
    for train_idx, val_idx in kf.split(X):
        # Slice train and validation sets for this fold
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Fit our pipeline on training data only to keep validation data completely unseen (prevents leakage)
        X_train_processed = pipeline.fit_transform(X_train)
        X_val_processed = pipeline.transform(X_val)
        
        # Log-transform target if selected (helps handle skewness)
        y_train_fit = np.log1p(y_train) if use_log_target else y_train
            
        # Fit the estimator
        model.fit(X_train_processed, y_train_fit)
        
        # Predict on validation data
        val_pred_fit = model.predict(X_val_processed)
        
        # Convert predictions back to dollar scale if we log-transformed the target
        val_pred = np.expm1(val_pred_fit) if use_log_target else val_pred_fit
            
        # Record scores
        fold_metrics = evaluate_predictions(y_val, val_pred)
        metrics_list.append(fold_metrics)
        
    # Calculate the average and standard deviation of metrics across folds
    agg_metrics = {}
    for key in metrics_list[0].keys():
        agg_metrics[f"CV {key} Mean"] = np.mean([m[key] for m in metrics_list])
        agg_metrics[f"CV {key} Std"] = np.std([m[key] for m in metrics_list])
        
    return agg_metrics

def train_tuned_models(X_train, y_train):
    """
    Defines our suite of models with cross-validated hyperparameter tuning.
    Contains OLS Baseline, L2 Ridge, L1 Lasso, and hybrid ElasticNet.
    """
    cv_strategy = KFold(n_splits=CROSS_VALIDATION_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    models = {
        "OLS Regression": LinearRegression(),
        "Ridge Regression": RidgeCV(alphas=ALPHA_GRID, cv=cv_strategy),
        "Lasso Regression": LassoCV(alphas=ALPHA_GRID, cv=cv_strategy, max_iter=10000, random_state=RANDOM_STATE),
        "ElasticNet Regression": ElasticNetCV(alphas=ALPHA_GRID, l1_ratio=L1_RATIOS, cv=cv_strategy, max_iter=10000, random_state=RANDOM_STATE)
    }
    
    return models
