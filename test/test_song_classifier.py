"""
Unit tests for the Song Genre Classification system.

This module contains comprehensive tests for all major components
of the song classification system.
"""

import pandas as pd
import pytest
from pathlib import Path


def test_config_fixture(config):
    """Test that the config fixture returns a valid configuration."""
    assert config is not None
    # Add more specific config tests based on your config structure


def test_temp_data_dir_fixture(temp_data_dir):
    """Test that the temp_data_dir fixture creates the expected files."""
    # Check that the directory exists
    assert temp_data_dir.exists()
    assert temp_data_dir.is_dir()
    
    # Check that CSV file exists and has correct content
    csv_file = temp_data_dir / "fma-rock-vs-hiphop.csv"
    assert csv_file.exists()
    
    csv_data = pd.read_csv(csv_file)
    assert len(csv_data) == 5
    assert "track_id" in csv_data.columns
    assert "genre_top" in csv_data.columns
    assert list(csv_data["genre_top"]) == ["Rock", "Hip-Hop", "Rock", "Hip-Hop", "Rock"]
    
    # Check that JSON file exists and has correct content
    json_file = temp_data_dir / "echonest-metrics.json"
    assert json_file.exists()
    
    json_data = pd.read_json(json_file)
    assert len(json_data) == 5
    assert "track_id" in json_data.columns
    assert "danceability" in json_data.columns
    assert "energy" in json_data.columns


def test_sample_data_fixture(sample_data):
    """Test that the sample_data fixture returns expected DataFrame structure."""
    assert isinstance(sample_data, pd.DataFrame)
    assert len(sample_data) == 5
    
    # Check columns
    expected_columns = ["track_id", "genre_top", "danceability", "energy", "speechiness"]
    for col in expected_columns:
        assert col in sample_data.columns
    
    # Check data types
    assert sample_data["track_id"].dtype in ["int64", "int32"]
    assert sample_data["genre_top"].dtype == "object"
    assert sample_data["danceability"].dtype in ["float64", "float32"]
    
    # Check genre values
    genres = sample_data["genre_top"].unique()
    assert "Rock" in genres
    assert "Hip-Hop" in genres


def test_data_consistency_between_fixtures(temp_data_dir, sample_data):
    """Test that data is consistent between different fixtures."""
    # Load CSV data from temp directory
    csv_file = temp_data_dir / "fma-rock-vs-hiphop.csv"
    csv_data = pd.read_csv(csv_file)
    
    # Check that track_ids match
    assert list(csv_data["track_id"]) == list(sample_data["track_id"])
    
    # Check that genres match
    assert list(csv_data["genre_top"]) == list(sample_data["genre_top"])


def test_data_values_range(sample_data):
    """Test that audio feature values are within expected ranges."""
    # Audio features should be between 0 and 1
    audio_features = ["danceability", "energy", "speechiness"]
    
    for feature in audio_features:
        assert sample_data[feature].min() >= 0.0
        assert sample_data[feature].max() <= 1.0


def test_no_missing_values(sample_data):
    """Test that there are no missing values in the sample data."""
    assert not sample_data.isnull().any().any()


def test_unique_track_ids(sample_data):
    """Test that all track IDs are unique."""
    assert sample_data["track_id"].nunique() == len(sample_data)