import os
import sys
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import KFold

# Ensure project root is in the Python search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import TRAIN_PATH, TARGET, RANDOM_STATE, ALPHA_GRID, L1_RATIOS, CROSS_VALIDATION_FOLDS
from src.modeling import build_preprocessing_pipeline

app = Flask(__name__)

# Global variables for models and pipeline
pipeline = None
models = {}
dataset_stats = {}

# Cross-Validation performance summary (pre-computed from notebook results)
MODEL_PERFORMANCE_METRICS = {
    "OLS Baseline": {
        "RMSE": 45948.64,
        "MAE": 33242.61,
        "R2": 0.6363,
        "RMSLE": 0.2494
    },
    "OLS (Log Target)": {
        "RMSE": 45861.21,
        "MAE": 31799.31,
        "R2": 0.6361,
        "RMSLE": 0.2369
    },
    "Ridge Regression": {
        "RMSE": 45906.99,
        "MAE": 31810.25,
        "R2": 0.6354,
        "RMSLE": 0.2370
    },
    "Lasso Regression": {
        "RMSE": 46457.14,
        "MAE": 31981.55,
        "R2": 0.6268,
        "RMSLE": 0.2379
    },
    "ElasticNet Regression": {
        "RMSE": 45944.65,
        "MAE": 31812.30,
        "R2": 0.6348,
        "RMSLE": 0.2370
    }
}

def train_valuation_models():
    global pipeline, models, dataset_stats
    print("Training Valuation Models...")
    
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"Training dataset not found at '{TRAIN_PATH}'. Run 'python src/download_data.py' first.")
        
    # 1. Load and clean dataset
    train_df = pd.read_csv(TRAIN_PATH)
    train_df = train_df[train_df['GrLivArea'] <= 4000].reset_index(drop=True)
    
    # 2. Extract stats for frontend boundaries
    dataset_stats = {
        "grLivArea": {
            "min": int(train_df["GrLivArea"].min()),
            "max": int(train_df["GrLivArea"].max()),
            "mean": float(train_df["GrLivArea"].mean())
        },
        "salePrice": {
            "min": float(train_df[TARGET].min()),
            "max": float(train_df[TARGET].max()),
            "mean": float(train_df[TARGET].mean())
        }
    }
    
    # 3. Fit preprocessing pipeline
    pipeline = build_preprocessing_pipeline()
    X_processed = pipeline.fit_transform(train_df)
    
    # Target values
    y_raw = train_df[TARGET]
    y_log = np.log1p(y_raw)
    
    # 4. Set up CV strategy for tuned models
    cv_strategy = KFold(n_splits=CROSS_VALIDATION_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    # 5. Train all five models
    print("Fitting OLS Baseline...")
    models["OLS Baseline"] = LinearRegression().fit(X_processed, y_raw)
    
    print("Fitting OLS (Log Target)...")
    models["OLS (Log Target)"] = LinearRegression().fit(X_processed, y_log)
    
    print("Fitting Ridge Regression...")
    models["Ridge Regression"] = RidgeCV(alphas=ALPHA_GRID, cv=cv_strategy).fit(X_processed, y_log)
    
    print("Fitting Lasso Regression...")
    models["Lasso Regression"] = LassoCV(alphas=ALPHA_GRID, cv=cv_strategy, max_iter=10000, random_state=RANDOM_STATE).fit(X_processed, y_log)
    
    print("Fitting ElasticNet Regression...")
    models["ElasticNet Regression"] = ElasticNetCV(alphas=ALPHA_GRID, l1_ratio=L1_RATIOS, cv=cv_strategy, max_iter=10000, random_state=RANDOM_STATE).fit(X_processed, y_log)
    
    print("All models successfully fitted and ready in memory.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/stats', methods=['GET'])
def get_stats():
    return jsonify(dataset_stats)

@app.route('/api/predict', methods=['POST'])
def predict():
    if not models or pipeline is None:
        return jsonify({"error": "Models not trained yet."}), 503
        
    try:
        data = request.get_json()
        
        # Build a single-row DataFrame matching training dataset columns
        input_data = pd.DataFrame([{
            "GrLivArea": float(data.get("grLivArea", 1500)),
            "BedroomAbvGr": float(data.get("bedroomAbvGr", 3)),
            "FullBath": float(data.get("fullBath", 2)),
            "HalfBath": float(data.get("halfBath", 1)),
            "BsmtFullBath": float(data.get("bsmtFullBath", 0)),
            "BsmtHalfBath": float(data.get("bsmtHalfBath", 0))
        }])
        
        # Transform inputs using the fitted preprocessing pipeline
        processed_input = pipeline.transform(input_data)
        
        # Query predictions from all models
        predictions = {}
        
        # OLS Baseline (predicts directly on dollar scale)
        predictions["OLS Baseline"] = float(models["OLS Baseline"].predict(processed_input)[0])
        
        # Log models (predict on log scale, require expm1 back-transform)
        for model_name in ["OLS (Log Target)", "Ridge Regression", "Lasso Regression", "ElasticNet Regression"]:
            pred_log = models[model_name].predict(processed_input)[0]
            predictions[model_name] = float(np.expm1(pred_log))
            
        return jsonify({
            "predictions": predictions,
            "metrics": MODEL_PERFORMANCE_METRICS
        })
        
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 400

@app.route('/api/coefficients', methods=['GET'])
def get_coefficients():
    if not models:
        return jsonify({"error": "Models not trained yet."}), 503
        
    try:
        features = ["Above Grade Living Area (GrLivArea)", "Bedrooms (BedroomAbvGr)", "Total Bathrooms (TotalBath)"]
        coefs = {}
        
        # Extract coefs (Note: OLS raw is on dollar scale, log models are on log-multiplier scale)
        for model_name, model_obj in models.items():
            coefs[model_name] = model_obj.coef_.tolist()
            
        return jsonify({
            "features": features,
            "coefficients": coefs
        })
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve coefficients: {str(e)}"}), 400

if __name__ == '__main__':
    # Train the models on startup
    train_valuation_models()
    # Run the server locally on port 5000
    app.run(host='127.0.0.1', port=5000, debug=True)
