"""
dataframe preprocessing file
"""

from typing import Any, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def scale_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    This function is scaling the features dataframe
    Arguments:
    ---------------------------------------------
    -dayaframe : The feautures dataframe

    Returns:
    ---------------------------------------------
    -dataframe : The scaled features dataframe
    """
    # Scale the features and set the values to a new variable
    scaler = StandardScaler()
    scaled_train_features = scaler.fit_transform(dataframe)
    return scaled_train_features
