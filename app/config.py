from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "graph_model.json"
PCA_PATH = MODEL_DIR / "pca_model.joblib"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.joblib"