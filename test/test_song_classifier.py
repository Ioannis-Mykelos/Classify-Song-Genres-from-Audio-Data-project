"""
Unit tests for the Song Genre Classification system.

This module contains comprehensive tests for all major components
of the song classification system.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from config import FEATURE_CONFIG, MODEL_CONFIG, get_config
from song_genre_classifier import SongGenreClassifier


class TestSongGenreClassifier:
    """Test cases for the SongGenreClassifier class."""

    def test_init(self, temp_data_dir):
        """Test classifier initialization."""
        classifier = SongGenreClassifier(data_dir=str(temp_data_dir))

        assert classifier.data_dir == Path(temp_data_dir)
        assert classifier.random_state == 10
        assert classifier.scaler is not None
        assert classifier.pca is None
        assert classifier.models == {}

    def test_init_nonexistent_dir(self):
        """Test initialization with non-existent directory."""
        with pytest.raises(FileNotFoundError):
            SongGenreClassifier(data_dir="nonexistent_dir")

    def test_load_data(self, classifier):
        """Test data loading functionality."""
        data = classifier.load_data()

        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert "genre_top" in data.columns
        assert "track_id" in data.columns
        assert "danceability" in data.columns

    def test_load_data_missing_files(self):
        """Test data loading with missing files."""
        temp_dir = Path(tempfile.mkdtemp())
        classifier = SongGenreClassifier(data_dir=str(temp_dir))

        with pytest.raises(FileNotFoundError):
            classifier.load_data()

        shutil.rmtree(temp_dir)

    def test_analyze_correlations(self, mock_matplotlib, classifier, sample_data):
        """Test correlation analysis."""
        corr_matrix = classifier.analyze_correlations(
            sample_data.drop(columns=["genre_top", "track_id"])
        )

        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape[0] == corr_matrix.shape[1]
        assert all(corr_matrix.index == corr_matrix.columns)

    def test_preprocess_data(self, classifier, sample_data):
        """Test data preprocessing."""
        scaled_features, labels = classifier.preprocess_data(sample_data)

        assert isinstance(scaled_features, np.ndarray)
        assert isinstance(labels, pd.Series)
        assert scaled_features.shape[0] == len(labels)
        assert (
            scaled_features.shape[1] == len(sample_data.columns) - 2
        )  # Exclude genre_top and track_id

    def test_perform_pca(self, mock_matplotlib, classifier, sample_data):
        """Test PCA functionality."""
        scaled_features, _ = classifier.preprocess_data(sample_data)
        pca_projection = classifier.perform_pca(scaled_features, n_components=3)

        assert isinstance(pca_projection, np.ndarray)
        assert pca_projection.shape[1] == 3
        assert classifier.pca is not None
        assert classifier.pca_projection is not None

    def test_balance_dataset(self, classifier, imbalanced_data):
        """Test dataset balancing."""
        balanced_data = classifier.balance_dataset(imbalanced_data)

        assert isinstance(balanced_data, pd.DataFrame)
        genre_counts = balanced_data["genre_top"].value_counts()
        assert len(genre_counts.unique()) == 1  # All classes should have same count
        assert all(count == genre_counts.min() for count in genre_counts.values)

    def test_train_models(self, mock_matplotlib, classifier, sample_data):
        """Test model training."""
        scaled_features, labels = classifier.preprocess_data(sample_data)
        pca_features = classifier.perform_pca(scaled_features, n_components=3)

        results = classifier.train_models(pca_features, labels)

        assert "models" in results
        assert "predictions" in results
        assert "test_labels" in results
        assert "decision_tree" in results["models"]
        assert "logistic_regression" in results["models"]
        assert len(results["predictions"]["tree"]) > 0
        assert len(results["predictions"]["logreg"]) > 0

    def test_cross_validate_models(self, mock_matplotlib, classifier, sample_data):
        """Test cross-validation."""
        scaled_features, labels = classifier.preprocess_data(sample_data)
        pca_features = classifier.perform_pca(scaled_features, n_components=3)

        cv_results = classifier.cross_validate_models(pca_features, labels, cv_folds=3)

        assert "decision_tree_mean" in cv_results
        assert "logistic_regression_mean" in cv_results
        assert "decision_tree_std" in cv_results
        assert "logistic_regression_std" in cv_results
        assert 0 <= cv_results["decision_tree_mean"] <= 1
        assert 0 <= cv_results["logistic_regression_mean"] <= 1

    def test_plot_confusion_matrices(self, mock_matplotlib, classifier, sample_data):
        """Test confusion matrix plotting."""
        scaled_features, labels = classifier.preprocess_data(sample_data)
        pca_features = classifier.perform_pca(scaled_features, n_components=3)
        results = classifier.train_models(pca_features, labels)

        # Should not raise any exceptions
        classifier.plot_confusion_matrices(results)

    def test_save_models(
        self, mock_matplotlib, classifier, sample_data, temp_model_dir
    ):
        """Test model saving."""
        scaled_features, labels = classifier.preprocess_data(sample_data)
        pca_features = classifier.perform_pca(scaled_features, n_components=3)
        classifier.train_models(pca_features, labels)

        classifier.save_models(temp_model_dir)

        assert (temp_model_dir / "decision_tree.pkl").exists()
        assert (temp_model_dir / "logistic_regression.pkl").exists()
        assert (temp_model_dir / "scaler.pkl").exists()
        assert (temp_model_dir / "pca.pkl").exists()

    def test_load_models(
        self, mock_matplotlib, classifier, sample_data, temp_model_dir
    ):
        """Test model loading."""
        scaled_features, labels = classifier.preprocess_data(sample_data)
        pca_features = classifier.perform_pca(scaled_features, n_components=3)
        classifier.train_models(pca_features, labels)
        classifier.save_models(temp_model_dir)

        # Create new classifier and load models
        new_classifier = SongGenreClassifier(data_dir=classifier.data_dir)
        new_classifier.load_models(temp_model_dir)

        assert len(new_classifier.models) > 0
        assert new_classifier.scaler is not None
        assert new_classifier.pca is not None

    def test_predict(self, mock_matplotlib, classifier, sample_data):
        """Test prediction functionality."""
        scaled_features, labels = classifier.preprocess_data(sample_data)
        pca_features = classifier.perform_pca(scaled_features, n_components=3)
        classifier.train_models(pca_features, labels)

        # Test prediction with sample features
        test_features = np.random.randn(2, 3)  # 2 samples, 3 features
        predictions = classifier.predict(test_features)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 2
        assert all(pred in ["Rock", "Hip-Hop"] for pred in predictions)

    def test_predict_invalid_model(self, mock_matplotlib, classifier, sample_data):
        """Test prediction with invalid model name."""
        scaled_features, labels = classifier.preprocess_data(sample_data)
        pca_features = classifier.perform_pca(scaled_features, n_components=3)
        classifier.train_models(pca_features, labels)

        test_features = np.random.randn(1, 3)

        with pytest.raises(ValueError):
            classifier.predict(test_features, model_name="invalid_model")

    def test_run_full_pipeline(self, mock_matplotlib, classifier):
        """Test the complete pipeline."""
        results = classifier.run_full_pipeline(balance_data=True, n_components=3)

        assert "correlation_matrix" in results
        assert "training_results" in results
        assert "cv_results" in results
        assert "data_info" in results
        assert results["data_info"]["features"] == 3
        assert "Rock" in results["data_info"]["genres"]
        assert "Hip-Hop" in results["data_info"]["genres"]

    def test_validate_data(self, classifier, sample_data):
        """Test data validation functionality."""
        # Test with valid data
        assert classifier.validate_data(sample_data) is True

    def test_validate_data_missing_columns(self, classifier):
        """Test data validation with missing required columns."""
        invalid_data = pd.DataFrame(
            {
                "track_id": [1, 2, 3],
                "danceability": [0.5, 0.8, 0.3],
            }
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            classifier.validate_data(invalid_data)

    def test_validate_data_empty_dataset(self, classifier):
        """Test data validation with empty dataset."""
        empty_data = pd.DataFrame()

        with pytest.raises(ValueError, match="Dataset is empty"):
            classifier.validate_data(empty_data)

    def test_predict_with_different_models(
        self, mock_matplotlib, classifier, sample_data
    ):
        """Test prediction with different model types."""
        scaled_features, labels = classifier.preprocess_data(sample_data)
        pca_features = classifier.perform_pca(scaled_features, n_components=3)
        classifier.train_models(pca_features, labels)

        test_features = np.random.randn(1, 3)

        # Test with decision tree
        tree_predictions = classifier.predict(test_features, model_name="decision_tree")
        assert isinstance(tree_predictions, np.ndarray)

        # Test with logistic regression
        logreg_predictions = classifier.predict(
            test_features, model_name="logistic_regression"
        )
        assert isinstance(logreg_predictions, np.ndarray)


class TestConfiguration:
    """Test cases for configuration management."""

    def test_get_config(self):
        """Test configuration retrieval."""
        config = get_config()

        assert isinstance(config, dict)
        assert "paths" in config
        assert "model" in config
        assert "features" in config
        assert "visualization" in config

    def test_model_config(self):
        """Test model configuration."""
        assert MODEL_CONFIG["random_state"] == 10
        assert MODEL_CONFIG["test_size"] == 0.2
        assert MODEL_CONFIG["cv_folds"] == 10
        assert MODEL_CONFIG["pca_components"] == 6

    def test_feature_config(self):
        """Test feature configuration."""
        assert FEATURE_CONFIG["target_column"] == "genre_top"
        assert FEATURE_CONFIG["id_column"] == "track_id"
        assert "Rock" in FEATURE_CONFIG["genres"]
        assert "Hip-Hop" in FEATURE_CONFIG["genres"]


class TestDataValidation:
    """Test cases for data validation."""

    def test_data_quality_checks(self):
        """Test data quality validation functions."""
        # Test with valid data
        valid_data = pd.DataFrame(
            {
                "track_id": [1, 2, 3],
                "genre_top": ["Rock", "Hip-Hop", "Rock"],
                "danceability": [0.5, 0.8, 0.3],
                "energy": [0.7, 0.6, 0.8],
            }
        )

        # Should not raise any exceptions
        assert len(valid_data) > 0
        assert not valid_data.isnull().all().any()
        assert valid_data["genre_top"].nunique() >= 2

    def test_missing_data_handling(self):
        """Test handling of missing data."""
        data_with_missing = pd.DataFrame(
            {
                "track_id": [1, 2, 3],
                "genre_top": ["Rock", None, "Hip-Hop"],
                "danceability": [0.5, 0.8, None],
                "energy": [0.7, None, 0.8],
            }
        )

        # Check for missing values
        assert data_with_missing.isnull().any().any()

        # Test data cleaning
        cleaned_data = data_with_missing.dropna()
        assert not cleaned_data.isnull().any().any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
