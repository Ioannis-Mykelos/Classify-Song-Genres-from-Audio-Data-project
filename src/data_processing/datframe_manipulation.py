"""
dataframe manipulation file
"""

from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd


def load_data() -> pd.DataFrame:
    """
    This function loads the two dataframes from the data
    folder and merges the two dataframes to one final.

    Arguments:
    ---------------------------------------------
    -None

    Returns:
    ---------------------------------------------
    -dataframe : The final dataframe we are going
                 to work on

    """
    # Read in track metadata with genre labels
    tracks = pd.read_csv("data/fma-rock-vs-hiphop.csv")
    # Read in track metrics with the features
    echonest_metrics = pd.read_json("data/echonest-metrics.json", precise_float=True)
    # Merge the relevant columns of tracks and echonest_metrics
    echo_tracks = echonest_metrics.merge(
        tracks[["genre_top", "track_id"]], on="track_id"
    )
    return echo_tracks


def split_features_n_labels(
    dataframe: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function splits the main df into features and labels dataframes

    Arguments:
    ---------------------------------------------
    -datadrame : The initial dataframe

    Returns:
    ---------------------------------------------
    -features : The features dataframe
    -labels   : The labels dataframe

    """
    # Define our features
    features = dataframe.drop(columns=["genre_top", "track_id"])
    # Define our labels
    labels = dataframe["genre_top"]

    return (features, labels)
