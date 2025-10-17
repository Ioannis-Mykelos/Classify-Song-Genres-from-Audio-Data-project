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

import sys

# Ensure repository root is on sys.path for CI environments
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_config


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
