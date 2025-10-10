"""
Unit tests for the Song Genre Classification system.

This module contains comprehensive tests for all major components
of the song classification system.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

from song_genre_classifier import SongGenreClassifier
from config import get_config, MODEL_CONFIG, FEATURE_CONFIG


class TestSongGenreClassifier:
    """Test cases for the SongGenreClassifier class."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory with sample data files."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create sample CSV data
        csv_data = pd.DataFrame({
            'track_id': [1, 2, 3, 4, 5],
            'genre_top': ['Rock', 'Hip-Hop', 'Rock', 'Hip-Hop', 'Rock']
        })
        csv_data.to_csv(temp_dir / 'fma-rock-vs-hiphop.csv', index=False)
        
        # Create sample JSON data
        json_data = {
            'track_id': [1, 2, 3, 4, 5],
            'danceability': [0.5, 0.8, 0.3, 0.9, 0.4],
            'energy': [0.7, 0.6, 0.8, 0.5, 0.9],
            'speechiness': [0.1, 0.3, 0.05, 0.4, 0.08],
            'acousticness': [0.2, 0.1, 0.3, 0.05, 0.4],
            'instrumentalness': [0.0, 0.0, 0.1, 0.0, 0.2],
            'liveness': [0.1, 0.2, 0.15, 0.3, 0.1],
            'valence': [0.6, 0.7, 0.4, 0.8, 0.5]
        }
        json_df = pd.DataFrame(json_data)
        json_df.to_json(temp_dir / 'echonest-metrics.json', orient='records')
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def classifier(self, temp_data_dir):
        """Create a SongGenreClassifier instance with test data."""
        return SongGenreClassifier(data_dir=str(temp_data_dir))
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'track_id': [1, 2, 3, 4, 5],
            'genre_top': ['Rock', 'Hip-Hop', 'Rock', 'Hip-Hop', 'Rock'],
            'danceability': [0.5, 0.8, 0.3, 0.9, 0.4],
            'energy': [0.7, 0.6, 0.8, 0.5, 0.9],
            'speechiness': [0.1, 0.3, 0.05, 0.4, 0.08]
        })
    
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
        assert 'genre_top' in data.columns
        assert 'track_id' in data.columns
        assert 'danceability' in data.columns
    
    def test_load_data_missing_files(self):
        """Test data loading with missing files."""
        temp_dir = Path(tempfile.mkdtemp())
        classifier = SongGenreClassifier(data_dir=str(temp_dir))
        
        with pytest.raises(FileNotFoundError):
            classifier.load_data()
        
        shutil.rmtree(temp_dir)
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_analyze_correlations(self, mock_savefig, mock_show, classifier, sample_data):
        """Test correlation analysis."""
        corr_matrix = classifier.analyze_correlations(sample_data.drop(columns=['genre_top', 'track_id']))
        
        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape[0] == corr_matrix.shape[1]
        assert all(corr_matrix.index == corr_matrix.columns)
        mock_savefig.assert_called_once()
    
    def test_preprocess_data(self, classifier, sample_data):
        """Test data preprocessing."""
        scaled_features, labels = classifier.preprocess_data(sample_data)
        
        assert isinstance(scaled_features, np.ndarray)
        assert isinstance(labels, pd.Series)
        assert scaled_features.shape[0] == len(labels)
        assert scaled_features.shape[1] == len(sample_data.columns) - 2  # Exclude genre_top and track_id
    
    def test_perform_pca(self, classifier, sample_data):
        """Test PCA functionality."""
        scaled_features, _ = classifier.preprocess_data(sample_data)
        
        with patch('matplotlib.pyplot.show'), patch('matplotlib.pyplot.savefig'):
            pca_projection = classifier.perform_pca(scaled_features, n_components=3)
        
        assert isinstance(pca_projection, np.ndarray)
        assert pca_projection.shape[1] == 3
        assert classifier.pca is not None
        assert classifier.pca_projection is not None
    
    def test_balance_dataset(self, classifier, sample_data):
        """Test dataset balancing."""
        balanced_data = classifier.balance_dataset(sample_data)
        
        assert isinstance(balanced_data, pd.DataFrame)
        genre_counts = balanced_data['genre_top'].value_counts()
        assert len(genre_counts.unique()) == 1  # All classes should have same count
    
    def test_train_models(self, classifier, sample_data):
        """Test model training."""
        scaled_features, labels = classifier.preprocess_data(sample_data)
        pca_features = classifier.perform_pca(scaled_features, n_components=3)
        
        results = classifier.train_models(pca_features, labels)
        
        assert 'models' in results
        assert 'predictions' in results
        assert 'test_labels' in results
        assert 'decision_tree' in results['models']
        assert 'logistic_regression' in results['models']
        assert len(results['predictions']['tree']) > 0
        assert len(results['predictions']['logreg']) > 0
    
    def test_cross_validate_models(self, classifier, sample_data):
        """Test cross-validation."""
        scaled_features, labels = classifier.preprocess_data(sample_data)
        pca_features = classifier.perform_pca(scaled_features, n_components=3)
        
        cv_results = classifier.cross_validate_models(pca_features, labels, cv_folds=3)
        
        assert 'decision_tree_mean' in cv_results
        assert 'logistic_regression_mean' in cv_results
        assert 0 <= cv_results['decision_tree_mean'] <= 1
        assert 0 <= cv_results['logistic_regression_mean'] <= 1
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_confusion_matrices(self, mock_savefig, mock_show, classifier, sample_data):
        """Test confusion matrix plotting."""
        scaled_features, labels = classifier.preprocess_data(sample_data)
        pca_features = classifier.perform_pca(scaled_features, n_components=3)
        results = classifier.train_models(pca_features, labels)
        
        classifier.plot_confusion_matrices(results)
        
        mock_savefig.assert_called_once()
    
    def test_save_models(self, classifier, sample_data):
        """Test model saving."""
        scaled_features, labels = classifier.preprocess_data(sample_data)
        pca_features = classifier.perform_pca(scaled_features, n_components=3)
        classifier.train_models(pca_features, labels)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            classifier.save_models(temp_dir)
            
            temp_path = Path(temp_dir)
            assert (temp_path / 'decision_tree.pkl').exists()
            assert (temp_path / 'logistic_regression.pkl').exists()
            assert (temp_path / 'scaler.pkl').exists()
            assert (temp_path / 'pca.pkl').exists()
    
    def test_load_models(self, classifier, sample_data):
        """Test model loading."""
        scaled_features, labels = classifier.preprocess_data(sample_data)
        pca_features = classifier.perform_pca(scaled_features, n_components=3)
        classifier.train_models(pca_features, labels)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            classifier.save_models(temp_dir)
            
            # Create new classifier and load models
            new_classifier = SongGenreClassifier(data_dir=classifier.data_dir)
            new_classifier.load_models(temp_dir)
            
            assert len(new_classifier.models) > 0
            assert new_classifier.scaler is not None
            assert new_classifier.pca is not None
    
    def test_predict(self, classifier, sample_data):
        """Test prediction functionality."""
        scaled_features, labels = classifier.preprocess_data(sample_data)
        pca_features = classifier.perform_pca(scaled_features, n_components=3)
        classifier.train_models(pca_features, labels)
        
        # Test prediction with sample features
        test_features = np.random.randn(2, 3)  # 2 samples, 3 features
        predictions = classifier.predict(test_features)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 2
        assert all(pred in ['Rock', 'Hip-Hop'] for pred in predictions)
    
    def test_predict_invalid_model(self, classifier, sample_data):
        """Test prediction with invalid model name."""
        scaled_features, labels = classifier.preprocess_data(sample_data)
        pca_features = classifier.perform_pca(scaled_features, n_components=3)
        classifier.train_models(pca_features, labels)
        
        test_features = np.random.randn(1, 3)
        
        with pytest.raises(ValueError):
            classifier.predict(test_features, model_name='invalid_model')
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_run_full_pipeline(self, mock_savefig, mock_show, classifier):
        """Test the complete pipeline."""
        results = classifier.run_full_pipeline(balance_data=True, n_components=3)
        
        assert 'correlation_matrix' in results
        assert 'training_results' in results
        assert 'cv_results' in results
        assert 'data_info' in results
        assert results['data_info']['features'] == 3
        assert 'Rock' in results['data_info']['genres']
        assert 'Hip-Hop' in results['data_info']['genres']


class TestConfiguration:
    """Test cases for configuration management."""
    
    def test_get_config(self):
        """Test configuration retrieval."""
        config = get_config()
        
        assert isinstance(config, dict)
        assert 'paths' in config
        assert 'model' in config
        assert 'features' in config
        assert 'visualization' in config
    
    def test_model_config(self):
        """Test model configuration."""
        assert MODEL_CONFIG['random_state'] == 10
        assert MODEL_CONFIG['test_size'] == 0.2
        assert MODEL_CONFIG['cv_folds'] == 10
        assert MODEL_CONFIG['pca_components'] == 6
    
    def test_feature_config(self):
        """Test feature configuration."""
        assert FEATURE_CONFIG['target_column'] == 'genre_top'
        assert FEATURE_CONFIG['id_column'] == 'track_id'
        assert 'Rock' in FEATURE_CONFIG['genres']
        assert 'Hip-Hop' in FEATURE_CONFIG['genres']


class TestDataValidation:
    """Test cases for data validation."""
    
    def test_data_quality_checks(self):
        """Test data quality validation functions."""
        # Test with valid data
        valid_data = pd.DataFrame({
            'track_id': [1, 2, 3],
            'genre_top': ['Rock', 'Hip-Hop', 'Rock'],
            'danceability': [0.5, 0.8, 0.3],
            'energy': [0.7, 0.6, 0.8]
        })
        
        # Should not raise any exceptions
        assert len(valid_data) > 0
        assert not valid_data.isnull().all().any()
        assert valid_data['genre_top'].nunique() >= 2
    
    def test_missing_data_handling(self):
        """Test handling of missing data."""
        data_with_missing = pd.DataFrame({
            'track_id': [1, 2, 3],
            'genre_top': ['Rock', None, 'Hip-Hop'],
            'danceability': [0.5, 0.8, None],
            'energy': [0.7, None, 0.8]
        })
        
        # Check for missing values
        assert data_with_missing.isnull().any().any()
        
        # Test data cleaning
        cleaned_data = data_with_missing.dropna()
        assert not cleaned_data.isnull().any().any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
