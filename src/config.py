import os

# Base Directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, "notebooks")

# File Paths
TRAIN_PATH = os.path.join(RAW_DATA_DIR, "train.csv")
TEST_PATH = os.path.join(RAW_DATA_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(PROCESSED_DATA_DIR, "submission.csv")

# Model Configuration
RANDOM_STATE = 42
TEST_SPLIT_SIZE = 0.2
CROSS_VALIDATION_FOLDS = 5

# Feature Engineering Settings
RAW_BATHROOM_FEATURES = [
    "FullBath",
    "HalfBath",
    "BsmtFullBath",
    "BsmtHalfBath"
]
BASE_FEATURES = [
    "GrLivArea",      # Square footage (above grade living area)
    "BedroomAbvGr",   # Bedrooms above grade
]
ALL_MODEL_FEATURES = BASE_FEATURES + ["TotalBath"]
TARGET = "SalePrice"

# Model Hyperparameters Search Space
ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 200.0, 500.0]
L1_RATIOS = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
