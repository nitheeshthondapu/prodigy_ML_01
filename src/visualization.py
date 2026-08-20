import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
from sklearn.inspection import permutation_importance

def plot_target_transformations(y, save_path=None):
    """
    Plots the target variable before and after the log-transform.
    This helps visualize why log-transforming is so useful for skewed data.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left Plot: Original heavily skewed target
    sns.histplot(y, kde=True, color="royalblue", ax=axes[0])
    axes[0].set_title(f"Original Target (Skew: {y.skew():.2f})", fontsize=14)
    axes[0].set_xlabel("Sale Price ($)")
    
    # Right Plot: Transformed near-normal target
    y_log = np.log1p(y)
    sns.histplot(y_log, kde=True, color="seagreen", ax=axes[1])
    axes[1].set_title(f"Log-Transformed Target (Skew: {y_log.skew():.2f})", fontsize=14)
    axes[1].set_xlabel("Log(Sale Price + 1)")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_regression_diagnostics(y_true, y_pred, save_path=None):
    """
    Generates a 3-panel regression diagnostic layout:
    1. Actual vs. Predicted values (checks overall prediction quality).
    2. Residual vs. Fitted values with a trendline (checks homoscedasticity).
    3. Normal Q-Q Plot of residuals (checks if errors are normally distributed).
    """
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5))
    
    # 1. Actual vs Predicted Scatter
    sns.scatterplot(x=y_true, y=y_pred, ax=axes[0], color="teal", alpha=0.6, edgecolor="w")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Fit")
    axes[0].set_title("Actual vs. Predicted Sale Price", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Actual Price ($)", fontsize=12)
    axes[0].set_ylabel("Predicted Price ($)", fontsize=12)
    axes[0].legend(loc="upper left")
    
    # 2. Residuals vs Fitted Scatter with a LOWESS smoothing line
    sns.regplot(
        x=y_pred, 
        y=residuals, 
        ax=axes[1], 
        scatter_kws={"alpha": 0.5, "color": "coral", "edgecolor": "w"},
        line_kws={"color": "red", "lw": 2},
        lowess=True
    )
    axes[1].axhline(0, color="black", linestyle="--", alpha=0.7)
    axes[1].set_title("Residuals vs. Fitted Values (Homoscedasticity)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Fitted Values ($)", fontsize=12)
    axes[1].set_ylabel("Residuals ($)", fontsize=12)
    
    # 3. Normal Q-Q Probability Plot
    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].get_lines()[0].set_color("purple")
    axes[2].get_lines()[0].set_alpha(0.5)
    axes[2].get_lines()[0].set_markeredgecolor("w")
    axes[2].get_lines()[1].set_color("red")
    axes[2].get_lines()[1].set_linewidth(2)
    axes[2].set_title("Normal Q-Q Plot (Residuals Normality)", fontsize=14, fontweight="bold")
    axes[2].set_xlabel("Theoretical Quantiles", fontsize=12)
    axes[2].set_ylabel("Ordered Values", fontsize=12)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_feature_importance(model, X, y, feature_names, save_path=None):
    """
    Plots Permutation Feature Importance.
    This calculates how much validation R2 drops when we shuffle each feature.
    """
    # Run permutation importance sequentially (avoids Windows process-spawning issues)
    result = permutation_importance(
        model, X, y, n_repeats=10, random_state=42, n_jobs=1
    )
    
    # Sort features by importance
    sorted_importances_idx = result.importances_mean.argsort()
    
    # Render boxplot
    plt.figure(figsize=(10, 6))
    plt.boxplot(
        result.importances[sorted_importances_idx].T,
        vert=False,
        tick_labels=[feature_names[i] for i in sorted_importances_idx],
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", color="blue"),
        medianprops=dict(color="red", lw=2)
    )
    plt.title("Permutation Feature Importance (Validation Set)", fontsize=14, fontweight="bold")
    plt.xlabel("Decrease in R2 Score", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
