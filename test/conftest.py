"""
Pytest configuration and shared fixtures for the Song Genre Classification tests.

This module contains common fixtures that can be used across all test modules
in the test directory.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import get_config
from song_genre_classifier import SongGenreClassifier


@pytest.fixture(scope="session")
def config():
    """Get the application configuration."""
    return get_config()


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory with sample data files."""
    temp_dir = Path(tempfile.mkdtemp())

    # Create sample CSV data
    csv_data = pd.DataFrame(
        {
            "track_id": [1, 2, 3, 4, 5],
            "genre_top": ["Rock", "Hip-Hop", "Rock", "Hip-Hop", "Rock"],
        }
    )
    csv_data.to_csv(temp_dir / "fma-rock-vs-hiphop.csv", index=False)

    # Create sample JSON data
    json_data = {
        "track_id": [1, 2, 3, 4, 5],
        "danceability": [0.5, 0.8, 0.3, 0.9, 0.4],
        "energy": [0.7, 0.6, 0.8, 0.5, 0.9],
        "speechiness": [0.1, 0.3, 0.05, 0.4, 0.08],
        "acousticness": [0.2, 0.1, 0.3, 0.05, 0.4],
        "instrumentalness": [0.0, 0.0, 0.1, 0.0, 0.2],
        "liveness": [0.1, 0.2, 0.15, 0.3, 0.1],
        "valence": [0.6, 0.7, 0.4, 0.8, 0.5],
    }
    json_df = pd.DataFrame(json_data)
    json_df.to_json(temp_dir / "echonest-metrics.json", orient="records")

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def classifier(temp_data_dir):
    """Create a SongGenreClassifier instance with test data."""
    return SongGenreClassifier(data_dir=str(temp_data_dir))


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame(
        {
            "track_id": [1, 2, 3, 4, 5],
            "genre_top": ["Rock", "Hip-Hop", "Rock", "Hip-Hop", "Rock"],
            "danceability": [0.5, 0.8, 0.3, 0.9, 0.4],
            "energy": [0.7, 0.6, 0.8, 0.5, 0.9],
            "speechiness": [0.1, 0.3, 0.05, 0.4, 0.08],
        }
    )


@pytest.fixture
def large_sample_data():
    """Create a larger sample dataset for more comprehensive testing."""
    np.random.seed(42)
    n_samples = 100
    
    return pd.DataFrame(
        {
            "track_id": range(1, n_samples + 1),
            "genre_top": np.random.choice(["Rock", "Hip-Hop"], n_samples),
            "danceability": np.random.uniform(0, 1, n_samples),
            "energy": np.random.uniform(0, 1, n_samples),
            "speechiness": np.random.uniform(0, 1, n_samples),
            "acousticness": np.random.uniform(0, 1, n_samples),
            "instrumentalness": np.random.uniform(0, 1, n_samples),
            "liveness": np.random.uniform(0, 1, n_samples),
            "valence": np.random.uniform(0, 1, n_samples),
        }
    )


@pytest.fixture
def imbalanced_data():
    """Create imbalanced dataset for testing balancing functionality."""
    return pd.DataFrame(
        {
            "track_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "genre_top": ["Rock"] * 7 + ["Hip-Hop"] * 3,  # Imbalanced
            "danceability": [0.5, 0.8, 0.3, 0.9, 0.4, 0.6, 0.7, 0.2, 0.8, 0.5],
            "energy": [0.7, 0.6, 0.8, 0.5, 0.9, 0.4, 0.6, 0.8, 0.3, 0.7],
            "speechiness": [0.1, 0.3, 0.05, 0.4, 0.08, 0.2, 0.15, 0.35, 0.25, 0.12],
        }
    )


@pytest.fixture
def temp_model_dir():
    """Create a temporary directory for model saving/loading tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_matplotlib():
    """Mock matplotlib functions to prevent plot display during tests."""
    with pytest.MonkeyPatch.context() as m:
        m.setattr("matplotlib.pyplot.show", lambda: None)
        m.setattr("matplotlib.pyplot.savefig", lambda *args, **kwargs: None)
        yield
