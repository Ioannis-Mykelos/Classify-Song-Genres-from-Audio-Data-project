"""
Configuration settings for the Song Genre Classification system.

This module contains all configuration parameters, file paths, and settings
used throughout the application.
"""

import os
from pathlib import Path
from typing import Any, Dict

# Base paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, OUTPUTS_DIR]:
    directory.mkdir(exist_ok=True)

# Data file paths
DATA_FILES = {
    "tracks": DATA_DIR / "fma-rock-vs-hiphop.csv",
    "metrics": DATA_DIR / "echonest-metrics.json",
}

# Model configuration
MODEL_CONFIG = {
    "random_state": 10,
    "test_size": 0.2,
    "cv_folds": 10,
    "pca_components": 6,
    "pca_variance_threshold": 0.85,
    "balance_dataset": True,
}

# Model parameters
MODEL_PARAMS = {
    "decision_tree": {
        "random_state": MODEL_CONFIG["random_state"],
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
    },
    "logistic_regression": {
        "random_state": MODEL_CONFIG["random_state"],
        "max_iter": 1000,
        "solver": "liblinear",
        "C": 1.0,
    },
}

# Feature configuration
FEATURE_CONFIG = {
    "target_column": "genre_top",
    "id_column": "track_id",
    "genres": ["Rock", "Hip-Hop"],
    "scale_features": True,
    "apply_pca": True,
}

# Visualization configuration
VISUALIZATION_CONFIG = {
    "figure_size": (12, 8),
    "dpi": 300,
    "style": "seaborn-v0_8",
    "color_palette": "husl",
    "save_plots": True,
    "plot_formats": ["png", "pdf"],
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": LOGS_DIR / "song_classifier.log",
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5,
}

# Output file names
OUTPUT_FILES = {
    "correlation_matrix": "correlation_matrix.png",
    "pca_analysis": "pca_analysis.png",
    "confusion_matrices": "confusion_matrices.png",
    "model_performance": "model_performance.png",
    "feature_importance": "feature_importance.png",
}

# Model file names
MODEL_FILES = {
    "decision_tree": "decision_tree.pkl",
    "logistic_regression": "logistic_regression.pkl",
    "scaler": "scaler.pkl",
    "pca": "pca.pkl",
}

# Validation thresholds
VALIDATION_THRESHOLDS = {
    "min_samples_per_class": 10,
    "min_accuracy": 0.5,
    "max_correlation": 0.95,
    "min_explained_variance": 0.8,
}

# Environment-specific settings
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

if ENVIRONMENT == "production":
    LOGGING_CONFIG["level"] = "WARNING"
    VISUALIZATION_CONFIG["save_plots"] = True
elif ENVIRONMENT == "testing":
    MODEL_CONFIG["cv_folds"] = 3
    MODEL_CONFIG["test_size"] = 0.3

# Complete configuration dictionary
CONFIG = {
    "paths": {
        "project_root": PROJECT_ROOT,
        "data_dir": DATA_DIR,
        "models_dir": MODELS_DIR,
        "logs_dir": LOGS_DIR,
        "outputs_dir": OUTPUTS_DIR,
        "data_files": DATA_FILES,
    },
    "model": MODEL_CONFIG,
    "model_params": MODEL_PARAMS,
    "features": FEATURE_CONFIG,
    "visualization": VISUALIZATION_CONFIG,
    "logging": LOGGING_CONFIG,
    "output_files": OUTPUT_FILES,
    "model_files": MODEL_FILES,
    "validation": VALIDATION_THRESHOLDS,
    "environment": ENVIRONMENT,
}


def get_config() -> Dict[str, Any]:
    """
    Get the complete configuration dictionary.

    Returns:
        Dictionary containing all configuration settings
    """
    return CONFIG


def get_data_paths() -> Dict[str, Path]:
    """
    Get data file paths.

    Returns:
        Dictionary mapping data file names to their paths
    """
    return DATA_FILES


def get_model_config() -> Dict[str, Any]:
    """
    Get model configuration settings.

    Returns:
        Dictionary containing model configuration
    """
    return MODEL_CONFIG


def get_feature_config() -> Dict[str, Any]:
    """
    Get feature configuration settings.

    Returns:
        Dictionary containing feature configuration
    """
    return FEATURE_CONFIG
