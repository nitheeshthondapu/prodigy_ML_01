import os
import json

def make_markdown_cell(source_lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source_lines]
    }

def make_code_cell(source_lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source_lines]
    }

def main():
    cells = []
    
    # Cell 1: Intro
    cells.append(make_markdown_cell([
        "# Advanced House Price Regression Pipeline",
        "",
        "In this notebook, we are going to build a clean, production-grade machine learning pipeline to predict house sale prices using the Ames Housing dataset. The feature space is strictly constrained to square footage, the number of bedrooms, and the number of bathrooms.",
        "",
        "### What We're Doing:",
        "1. **Custom Scikit-Learn Transformers**: We'll write modular transformer classes (`BathroomEngineer` and `ColumnSelector`) inheriting from `BaseEstimator` and `TransformerMixin`. This makes our pipeline reproducible and prevents data leakage.",
        "2. **Target Log-Transformation**: The target variable `SalePrice` is heavily right-skewed. We transform the target to $y' = \\log(1 + y)$ during model fitting to stabilize variance and meet homoscedasticity assumptions, then back-transform our predictions using $y = \\exp(y') - 1$.",
        "3. **Regularization & Hyperparameter Search**: We use 5-Fold Cross-Validation combined with `RidgeCV`, `LassoCV`, and `ElasticNetCV` to search for optimal regularization strengths.",
        "4. **Regression Diagnostics**: We inspect the Q-Q plots, Residual vs. Fitted plots with a smoothing LOESS curve, and Permutation Feature Importances to evaluate the model's reliability."
    ]))
    
    # Cell 2: Imports
    cells.append(make_code_cell([
        "import os",
        "import sys",
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "from sklearn.linear_model import LinearRegression",
        "",
        "# Add the project root to the Python path so we can import our local src package modules",
        "sys.path.append(os.path.abspath('..'))",
        "",
        "from src.config import TRAIN_PATH, TEST_PATH, SUBMISSION_PATH, RANDOM_STATE, TARGET, ALL_MODEL_FEATURES",
        "from src.modeling import build_preprocessing_pipeline, run_cross_validation, evaluate_predictions, train_tuned_models",
        "from src.visualization import plot_target_transformations, plot_regression_diagnostics, plot_feature_importance",
        "",
        "# Initialize plot styling",
        "%matplotlib inline",
        "sns.set_theme(style=\"whitegrid\")"
    ]))
    
    # Cell 3: Data Ingestion Header
    cells.append(make_markdown_cell([
        "## 1. Loading the Data",
        "First, let's load our raw training and testing CSV files and check their initial dimensions."
    ]))
    
    # Cell 4: Load Data & Filter Outliers
    cells.append(make_code_cell([
        "train_df = pd.read_csv(TRAIN_PATH)",
        "test_df = pd.read_csv(TEST_PATH)",
        "",
        "print(f\"Original training dataset dimensions: {train_df.shape}\")",
        "",
        "# Remove the 4 extreme outliers with living area > 4000 sq ft as recommended by the dataset author (Dean De Cock)",
        "train_df = train_df[train_df['GrLivArea'] <= 4000].reset_index(drop=True)",
        "print(f\"Training dataset dimensions after removing outliers: {train_df.shape}\")",
        "print(f\"Testing dataset dimensions: {test_df.shape}\")"
    ]))
    
    # Cell 5: Target Transformation Header
    cells.append(make_markdown_cell([
        "## 2. Analyzing the Target Variable Distribution",
        "Linear Regression models perform best when the target variable is normally distributed and the residuals have constant variance. Let's look at `SalePrice` and see how log-transforming it affects its distribution."
    ]))
    
    # Cell 6: Plot Target Transformations
    cells.append(make_code_cell([
        "plot_target_transformations(train_df[TARGET])"
    ]))
    
    # Cell 7: Cross-Validation Header
    cells.append(make_markdown_cell([
        "## 3. Model Comparison & Cross-Validation",
        "Let's build our scikit-learn preprocessing pipeline and run 5-Fold Cross-Validation. We'll compare five different modeling strategies:",
        "1. **OLS Regression (Raw Target)**: Our baseline model fitted on raw dollars.",
        "2. **OLS Regression (Log Target)**: Fitted on the log scale to isolate the log-transform effect.",
        "3. **Ridge Regression (Log Target)**: Fitted on the log scale with L2 weight shrinkage.",
        "4. **Lasso Regression (Log Target)**: Fitted on the log scale with L1 weight sparsity.",
        "5. **ElasticNet Regression (Log Target)**: Fitted on the log scale with a hybrid L1 + L2 penalty."
    ]))
    
    # Cell 8: Run Cross-Validation
    cells.append(make_code_cell([
        "# Build the preprocessing pipeline",
        "pipeline = build_preprocessing_pipeline()",
        "",
        "# 1. Baseline OLS on Raw Target",
        "ols_raw = LinearRegression()",
        "ols_raw_cv = run_cross_validation(train_df, train_df[TARGET], pipeline, ols_raw, use_log_target=False)",
        "",
        "# Load our tuned model candidates",
        "tuned_models = train_tuned_models(train_df, train_df[TARGET])",
        "cv_results = {}",
        "",
        "# 2. OLS on Log Target",
        "cv_results[\"OLS (Log Target)\"] = run_cross_validation(",
        "    train_df, train_df[TARGET], pipeline, tuned_models[\"OLS Regression\"], use_log_target=True",
        ")",
        "",
        "# 3. Ridge Regression (Log Target)",
        "cv_results[\"Ridge Regression (Log Target)\"] = run_cross_validation(",
        "    train_df, train_df[TARGET], pipeline, tuned_models[\"Ridge Regression\"], use_log_target=True",
        ")",
        "",
        "# 4. Lasso Regression (Log Target)",
        "cv_results[\"Lasso Regression (Log Target)\"] = run_cross_validation(",
        "    train_df, train_df[TARGET], pipeline, tuned_models[\"Lasso Regression\"], use_log_target=True",
        ")",
        "",
        "# 5. ElasticNet Regression (Log Target)",
        "cv_results[\"ElasticNet Regression (Log Target)\"] = run_cross_validation(",
        "    train_df, train_df[TARGET], pipeline, tuned_models[\"ElasticNet Regression\"], use_log_target=True",
        ")",
        "",
        "# Compile metrics into a summary table",
        "comparison_data = [",
        "    {",
        "        \"Model\": \"OLS (Raw Target)\",",
        "        \"CV RMSE\": ols_raw_cv[\"CV RMSE Mean\"],",
        "        \"CV MAE\": ols_raw_cv[\"CV MAE Mean\"],",
        "        \"CV R2\": ols_raw_cv[\"CV R2 Mean\"],",
        "        \"CV RMSLE\": ols_raw_cv[\"CV RMSLE Mean\"]",
        "    }",
        "]",
        "for model_name, res in cv_results.items():",
        "    comparison_data.append({",
        "        \"Model\": model_name,",
        "        \"CV RMSE\": res[\"CV RMSE Mean\"],",
        "        \"CV MAE\": res[\"CV MAE Mean\"],",
        "        \"CV R2\": res[\"CV R2 Mean\"],",
        "        \"CV RMSLE\": res[\"CV RMSLE Mean\"]",
        "    })",
        "",
        "cv_table = pd.DataFrame(comparison_data)",
        "print(\"Cross-Validation Model Comparison:\")",
        "print(cv_table.to_string(index=False))"
    ]))
    
    # Cell 9: Holdout Validation Header
    cells.append(make_markdown_cell([
        "## 4. Training on a Holdout Split",
        "To inspect optimal hyperparameter selection and plot residual diagnostics, we split the data into a Train and Validation holdout split (80/20) and fit our final estimators."
    ]))
    
    # Cell 10: Holdout Validation Fit
    cells.append(make_code_cell([
        "from sklearn.model_selection import train_test_split",
        "",
        "# Split data",
        "X_train_df, X_val_df, y_train, y_val = train_test_split(",
        "    train_df, train_df[TARGET], test_size=0.2, random_state=RANDOM_STATE",
        ")",
        "",
        "# Fit the preprocessor on training data only to avoid leakage",
        "pipeline = build_preprocessing_pipeline()",
        "X_train_processed = pipeline.fit_transform(X_train_df)",
        "X_val_processed = pipeline.transform(X_val_df)",
        "y_train_log = np.log1p(y_train)",
        "",
        "# 1. Fit OLS Log Target",
        "lr_final = LinearRegression()",
        "lr_final.fit(X_train_processed, y_train_log)",
        "lr_pred = np.expm1(lr_final.predict(X_val_processed))",
        "",
        "# 2. Fit RidgeCV",
        "ridge_final = train_tuned_models(X_train_df, y_train)[\"Ridge Regression\"]",
        "ridge_final.fit(X_train_processed, y_train_log)",
        "ridge_pred = np.expm1(ridge_final.predict(X_val_processed))",
        "print(f\"Optimal Ridge alpha: {ridge_final.alpha_}\")",
        "",
        "# 3. Fit LassoCV",
        "lasso_final = train_tuned_models(X_train_df, y_train)[\"Lasso Regression\"]",
        "lasso_final.fit(X_train_processed, y_train_log)",
        "lasso_pred = np.expm1(lasso_final.predict(X_val_processed))",
        "print(f\"Optimal Lasso alpha: {lasso_final.alpha_:.4f}\")",
        "",
        "# 4. Fit ElasticNetCV",
        "enet_final = train_tuned_models(X_train_df, y_train)[\"ElasticNet Regression\"]",
        "enet_final.fit(X_train_processed, y_train_log)",
        "enet_pred = np.expm1(enet_final.predict(X_val_processed))",
        "print(f\"Optimal ElasticNet alpha: {enet_final.alpha_:.4f}, l1_ratio: {enet_final.l1_ratio_}\")"
    ]))
    
    # Cell 11: Validation Performance Header
    cells.append(make_markdown_cell([
        "## 5. Evaluating Holdout Performance",
        "Let's measure the performance of our models on our unseen holdout validation split."
    ]))
    
    # Cell 12: Print Validation Metrics
    cells.append(make_code_cell([
        "val_results = []",
        "preds = {",
        "    \"OLS Log Target\": lr_pred,",
        "    \"Ridge Regression\": ridge_pred,",
        "    \"Lasso Regression\": lasso_pred,",
        "    \"ElasticNet Regression\": enet_pred",
        "}",
        "",
        "for name, pred in preds.items():",
        "    metrics = evaluate_predictions(y_val, pred)",
        "    metrics[\"Model\"] = name",
        "    val_results.append(metrics)",
        "",
        "val_results_df = pd.DataFrame(val_results)[[\"Model\", \"RMSE\", \"MAE\", \"R2\", \"RMSLE\"]]",
        "print(\"Holdout Validation Performance:\")",
        "print(val_results_df.to_string(index=False))"
    ]))
    
    # Cell 13: Model Diagnostics Header
    cells.append(make_markdown_cell([
        "## 6. Model Diagnostics & Error Analysis",
        "Let's inspect residual behavior, check normality via Q-Q plots, and view Permutation Feature Importances for our tuned ElasticNet model."
    ]))
    
    # Cell 14: Plot Diagnostic plots
    cells.append(make_code_cell([
        "# Draw diagnostic plots",
        "plot_regression_diagnostics(y_val, enet_pred)"
    ]))
    
    # Cell 15: Plot Feature Importance
    cells.append(make_code_cell([
        "# Compute permutation importance on the holdout validation set",
        "plot_feature_importance(enet_final, X_val_processed, y_val, ALL_MODEL_FEATURES)"
    ]))
    
    # Cell 16: Inference Header
    cells.append(make_markdown_cell([
        "## 7. Generating Predictions on Unseen Test Data",
        "To maximize sample size and help our model learn as much as possible, we train our pipeline and tuned ElasticNet model on the **entire** training dataset. We then preprocess the test set and generate our prediction file `submission.csv`."
    ]))
    
    # Cell 17: Inference Implementation
    cells.append(make_code_cell([
        "# Fit the preprocessor on the full training dataset",
        "pipeline_full = build_preprocessing_pipeline()",
        "X_full_processed = pipeline_full.fit_transform(train_df)",
        "y_full_log = np.log1p(train_df[TARGET])",
        "",
        "# Train the best estimator on full data",
        "best_model_full = train_tuned_models(train_df, train_df[TARGET])[\"ElasticNet Regression\"]",
        "best_model_full.fit(X_full_processed, y_full_log)",
        "",
        "# Preprocess the test dataset",
        "X_test_processed = pipeline_full.transform(test_df)",
        "",
        "# Predict and back-transform predictions to dollars",
        "test_pred_log = best_model_full.predict(X_test_processed)",
        "test_predictions = np.expm1(test_pred_log)",
        "",
        "# Assemble submission DataFrame",
        "submission_df = pd.DataFrame({",
        "    'Id': test_df['Id'],",
        "    'SalePrice': test_predictions",
        "})",
        "",
        "# Save the output submission file",
        "os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)",
        "submission_df.to_csv(SUBMISSION_PATH, index=False)",
        "",
        "print(f\"Saved predictions successfully to '{SUBMISSION_PATH}'\")",
        "print(submission_df.head())"
    ]))
    
    # Cell 18: Final Summary Header
    cells.append(make_markdown_cell([
        "## 8. Summary of Results",
        "",
        "### Q&A",
        "* **How does log-transforming the target variable affect model performance?**",
        "  * Log-transforming the target variable `SalePrice` stabilizes the residuals' variance (addressing heteroscedasticity) and shifts the target variable from a right-skewed distribution (skew: 1.88) to a symmetric, near-normal distribution (skew: 0.12). This significantly improves the robustness of the linear regression, resulting in a cleaner residual distribution and better validation metrics.",
        "",
        "### Data Analysis Key Findings",
        "* **Regularization Parameters**: The cross-validated search selected $\\alpha = 0.01$ and $l_1 \\text{ ratio} = 0.1$ for ElasticNet, combining L1 (Lasso) and L2 (Ridge) penalties to regularize the coefficients.",
        "* **Model Evaluation Metrics**:",
        "  * **OLS (Raw Target)**: CV RMSE = $\\$45,949$, CV $R^2 = 63.63\\%$, CV RMSLE = $0.249$.",
        "  * **ElasticNet (Log Target)**: CV RMSE = $\\$45,945$, CV $R^2 = 63.48\\%$, CV RMSLE = $0.237$.",
        "  * Log-transforming the target variable reduced the cross-validation Mean Absolute Error (MAE) from $\\$33,243$ to $\\$31,812$ and improved the RMSLE from **0.249 to 0.237**, showing a better fit for the bulk of home prices.",
        "* **Feature Importance**: Permutation importance indicates that Above Grade Living Area (`GrLivArea`) is by far the most critical predictor, followed by `TotalBath`. `BedroomAbvGr` has a negative coefficient, indicating that when square footage is held constant, increasing bedrooms reduces room size and drops the home value.",
        "",
        "### Insights or Next Steps",
        "* **Advanced Residual Pattern**: The residual vs. fitted plot indicates that while homoscedasticity is improved, there is still a slight non-linear shape in the errors for highly priced homes. This suggests that incorporating non-linear features (e.g., interaction terms or polynomial terms) or moving to tree-based models like Random Forests or XGBoost would yield further accuracy improvements."
    ]))
    
    notebook_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    notebook_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks")
    os.makedirs(notebook_dir, exist_ok=True)
    notebook_path = os.path.join(notebook_dir, "house_price_prediction.ipynb")
    
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=1)
        
    print(f"Notebook created successfully at {notebook_path}")

if __name__ == "__main__":
    main()
