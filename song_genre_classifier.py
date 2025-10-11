#!/usr/bin/env python3
"""
Song Genre Classification System

A machine learning system for classifying songs into Rock and Hip-Hop genres
using audio features from The Echo Nest dataset.

Author: [Your Name]
Date: 2024
License: MIT
"""

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from config import get_config, get_data_paths, get_feature_config, get_model_config

# Get configuration
CONFIG = get_config()

# Configure logging
logging.basicConfig(
    level=getattr(logging, CONFIG["logging"]["level"]),
    format=CONFIG["logging"]["format"],
    handlers=[
        logging.FileHandler(CONFIG["logging"]["log_file"]),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set matplotlib style
plt.style.use(CONFIG["visualization"]["style"])
sns.set_palette(CONFIG["visualization"]["color_palette"])


class SongGenreClassifier:
    """
    A machine learning system for classifying songs into Rock and Hip-Hop genres.

    This class handles data loading, preprocessing, feature engineering, model training,
    and evaluation for song genre classification.
    """

    def __init__(self, data_dir: str = "data", random_state: int = 10):
        """
        Initialize the SongGenreClassifier.

        Args:
            data_dir: Directory containing the dataset files
            random_state: Random state for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca: Optional[PCA] = None
        self.models: Dict[str, Any] = {}
        self.features: Optional[pd.DataFrame] = None
        self.labels: Optional[pd.Series] = None
        self.pca_projection: Optional[np.ndarray] = None

        # Ensure data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate the loaded dataset for quality and completeness.

        Args:
            data: DataFrame to validate

        Returns:
            True if data passes validation, False otherwise

        Raises:
            ValueError: If data fails validation checks
        """
        logger.info("Validating dataset...")

        # Check for required columns
        required_columns = ["track_id", "genre_top"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Check for empty dataset
        if len(data) == 0:
            raise ValueError("Dataset is empty")

        # Check for missing values in critical columns
        critical_columns = ["track_id", "genre_top"]
        for col in critical_columns:
            if data[col].isnull().any():
                logger.warning(f"Missing values found in column: {col}")
                # Remove rows with missing critical values
                data = data.dropna(subset=[col])

        # Check genre distribution
        genre_counts = data["genre_top"].value_counts()
        min_samples = CONFIG["validation"]["min_samples_per_class"]

        for genre, count in genre_counts.items():
            if count < min_samples:
                logger.warning(
                    f"Genre '{genre}' has only {count} samples (minimum: {min_samples})"
                )

        # Check for duplicate track_ids
        duplicates = data["track_id"].duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate track_ids")
            data = data.drop_duplicates(subset=["track_id"])

        # Check feature columns
        feature_columns = [
            col for col in data.columns if col not in ["track_id", "genre_top"]
        ]
        if len(feature_columns) == 0:
            raise ValueError("No feature columns found in dataset")

        # Check for constant features
        constant_features = []
        for col in feature_columns:
            if data[col].nunique() <= 1:
                constant_features.append(col)

        if constant_features:
            logger.warning(f"Constant features found: {constant_features}")

        logger.info("Data validation completed successfully")
        return True

    def load_data(self) -> pd.DataFrame:
        """
        Load and merge the song genre dataset.

        Returns:
            Merged DataFrame containing features and labels

        Raises:
            FileNotFoundError: If required data files are not found
            ValueError: If data validation fails
        """
        try:
            # Get file paths from configuration
            data_files = get_data_paths()
            tracks_file = data_files["tracks"]
            metrics_file = data_files["metrics"]

            # Check if files exist
            if not tracks_file.exists():
                raise FileNotFoundError(f"Tracks file not found: {tracks_file}")
            if not metrics_file.exists():
                raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

            logger.info("Loading dataset files...")

            # Read in track metadata with genre labels
            tracks = pd.read_csv(tracks_file)
            logger.info(f"Loaded {len(tracks)} tracks from CSV")

            # Read in track metrics with the features
            echonest_metrics = pd.read_json(metrics_file, precise_float=True)
            logger.info(f"Loaded {len(echonest_metrics)} track metrics from JSON")

            # Merge the relevant columns
            echo_tracks = echonest_metrics.merge(
                tracks[["genre_top", "track_id"]], on="track_id"
            )

            # Validate the merged data
            self.validate_data(echo_tracks)

            logger.info(f"Merged dataset contains {len(echo_tracks)} samples")
            logger.info(
                f"Genre distribution:\n{echo_tracks['genre_top'].value_counts()}"
            )

            return echo_tracks

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def analyze_correlations(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze correlations between features.

        Args:
            data: DataFrame containing the features

        Returns:
            Correlation matrix
        """
        logger.info("Analyzing feature correlations...")

        # Create correlation matrix
        corr_metrics = data.corr()

        # Create correlation heatmap
        fig_size = CONFIG["visualization"]["figure_size"]
        plt.figure(figsize=fig_size)
        mask = np.triu(np.ones_like(corr_metrics, dtype=bool))

        sns.heatmap(
            corr_metrics,
            mask=mask,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            fmt=".2f",
            cbar_kws={"shrink": 0.8},
        )
        plt.title("Feature Correlation Matrix", fontsize=16, fontweight="bold")
        plt.xlabel("Features", fontsize=12)
        plt.ylabel("Features", fontsize=12)
        plt.tight_layout()

        # Save plot if configured
        if CONFIG["visualization"]["save_plots"]:
            output_path = (
                CONFIG["paths"]["outputs_dir"]
                / CONFIG["output_files"]["correlation_matrix"]
            )
            plt.savefig(
                output_path, dpi=CONFIG["visualization"]["dpi"], bbox_inches="tight"
            )
            logger.info(f"Correlation matrix saved to: {output_path}")

        plt.show()

        return corr_metrics

    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, pd.Series]:
        """
        Preprocess the data by scaling features.

        Args:
            data: Raw dataset

        Returns:
            Tuple of (scaled_features, labels)
        """
        logger.info("Preprocessing data...")

        # Define features and labels
        self.features = data.drop(columns=["genre_top", "track_id"])
        self.labels = data["genre_top"]

        # Scale the features
        scaled_features = self.scaler.fit_transform(self.features)

        if self.features is not None and self.labels is not None:
            logger.info(
                f"Scaled {self.features.shape[1]} features for {len(self.labels)} samples"
            )

        return scaled_features, self.labels

    def perform_pca(
        self, scaled_features: np.ndarray, n_components: int = 6
    ) -> np.ndarray:
        """
        Perform Principal Component Analysis on scaled features.

        Args:
            scaled_features: Scaled feature matrix
            n_components: Number of principal components to use

        Returns:
            PCA projection of the data
        """
        logger.info(f"Performing PCA with {n_components} components...")

        # Fit PCA
        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        pca_projection = self.pca.fit_transform(scaled_features)

        # Calculate explained variance
        assert self.pca is not None  # Type assertion for mypy
        exp_variance = self.pca.explained_variance_ratio_
        cum_exp_variance = np.cumsum(exp_variance)

        logger.info(f"Explained variance by components: {exp_variance}")
        logger.info(f"Cumulative explained variance: {cum_exp_variance[-1]:.3f}")

        # Create visualization
        self._plot_pca_results(exp_variance, cum_exp_variance)

        self.pca_projection = pca_projection
        return pca_projection

    def _plot_pca_results(self, exp_variance: np.ndarray, cum_exp_variance: np.ndarray):
        """Create PCA visualization plots."""
        fig_size = CONFIG["visualization"]["figure_size"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_size[0] * 1.5, fig_size[1]))

        # Scree plot
        bars = ax1.bar(
            range(len(exp_variance)),
            exp_variance,
            color=sns.color_palette(CONFIG["visualization"]["color_palette"]),
        )
        ax1.set_xlabel("Principal Component #", fontsize=12)
        ax1.set_ylabel("Explained Variance Ratio", fontsize=12)
        ax1.set_title("PCA Scree Plot", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        # Cumulative explained variance
        ax2.plot(cum_exp_variance, marker="o", linewidth=2, markersize=6)
        ax2.axhline(
            y=CONFIG["model"]["pca_variance_threshold"],
            linestyle="--",
            color="red",
            alpha=0.7,
            linewidth=2,
        )
        ax2.set_xlabel("Number of Components", fontsize=12)
        ax2.set_ylabel("Cumulative Explained Variance", fontsize=12)
        ax2.set_title("Cumulative Explained Variance", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.1)

        # Add threshold label
        ax2.text(
            len(exp_variance) * 0.7,
            CONFIG["model"]["pca_variance_threshold"] + 0.05,
            f'{CONFIG["model"]["pca_variance_threshold"]*100}% threshold',
            fontsize=10,
            color="red",
            fontweight="bold",
        )

        plt.tight_layout()

        # Save plot if configured
        if CONFIG["visualization"]["save_plots"]:
            output_path = (
                CONFIG["paths"]["outputs_dir"] / CONFIG["output_files"]["pca_analysis"]
            )
            plt.savefig(
                output_path, dpi=CONFIG["visualization"]["dpi"], bbox_inches="tight"
            )
            logger.info(f"PCA analysis plot saved to: {output_path}")

        plt.show()

    def balance_dataset(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Balance the dataset by sampling equal numbers of Rock and Hip-Hop songs.

        Args:
            data: Original dataset

        Returns:
            Balanced dataset
        """
        logger.info("Balancing dataset...")

        # Get genre counts
        genre_counts = data["genre_top"].value_counts()
        logger.info(f"Original genre distribution: {genre_counts.to_dict()}")

        # Find the minority class size
        min_class_size = genre_counts.min()

        # Sample balanced data
        balanced_data = []
        for genre in data["genre_top"].unique():
            genre_data = data[data["genre_top"] == genre]
            if len(genre_data) > min_class_size:
                sampled_data = genre_data.sample(
                    min_class_size, random_state=self.random_state
                )
            else:
                sampled_data = genre_data
            balanced_data.append(sampled_data)

        balanced_df = pd.concat(balanced_data, ignore_index=True)

        logger.info(f"Balanced dataset size: {len(balanced_df)}")
        logger.info(
            f"Balanced genre distribution:\n{balanced_df['genre_top'].value_counts()}"
        )

        return balanced_df

    def train_models(self, features: np.ndarray, labels: pd.Series) -> Dict[str, Any]:
        """
        Train decision tree and logistic regression models.

        Args:
            features: Feature matrix
            labels: Target labels

        Returns:
            Dictionary containing trained models and their predictions
        """
        logger.info("Training models...")

        # Split data
        train_features, test_features, train_labels, test_labels = train_test_split(
            features, labels, random_state=self.random_state, test_size=0.2
        )

        # Train Decision Tree
        tree = DecisionTreeClassifier(random_state=self.random_state)
        tree.fit(train_features, train_labels)
        tree_pred = tree.predict(test_features)

        # Train Logistic Regression
        logreg = LogisticRegression(random_state=self.random_state, max_iter=1000)
        logreg.fit(train_features, train_labels)
        logreg_pred = logreg.predict(test_features)

        # Store models
        self.models = {"decision_tree": tree, "logistic_regression": logreg}

        # Evaluate models
        results = {
            "models": self.models,
            "predictions": {"tree": tree_pred, "logreg": logreg_pred},
            "test_labels": test_labels,
            "test_features": test_features,
        }

        # Print classification reports
        logger.info("Model Performance:")
        logger.info("Decision Tree:")
        logger.info(f"\n{classification_report(test_labels, tree_pred)}")
        logger.info("Logistic Regression:")
        logger.info(f"\n{classification_report(test_labels, logreg_pred)}")

        return results

    def cross_validate_models(
        self, features: np.ndarray, labels: pd.Series, cv_folds: int = 10
    ) -> Dict[str, float]:
        """
        Perform cross-validation on the models.

        Args:
            features: Feature matrix
            labels: Target labels
            cv_folds: Number of cross-validation folds

        Returns:
            Dictionary containing mean CV scores for each model
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation...")

        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        # Cross-validate Decision Tree
        tree = DecisionTreeClassifier(random_state=self.random_state)
        tree_scores = cross_val_score(tree, features, labels, cv=kf, scoring="accuracy")

        # Cross-validate Logistic Regression
        logreg = LogisticRegression(random_state=self.random_state, max_iter=1000)
        logreg_scores = cross_val_score(
            logreg, features, labels, cv=kf, scoring="accuracy"
        )

        cv_results = {
            "decision_tree_mean": np.mean(tree_scores),
            "decision_tree_std": np.std(tree_scores),
            "logistic_regression_mean": np.mean(logreg_scores),
            "logistic_regression_std": np.std(logreg_scores),
        }

        logger.info("Cross-validation results:")
        logger.info(
            f"Decision Tree: {cv_results['decision_tree_mean']:.3f} ± {cv_results['decision_tree_std']:.3f}"
        )
        logger.info(
            f"Logistic Regression: {cv_results['logistic_regression_mean']:.3f} ± {cv_results['logistic_regression_std']:.3f}"
        )

        return cv_results

    def plot_confusion_matrices(self, results: Dict[str, Any]):
        """
        Plot confusion matrices for both models.

        Args:
            results: Dictionary containing model results
        """
        fig_size = CONFIG["visualization"]["figure_size"]
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(fig_size[0] * 1.2, fig_size[1] * 0.8)
        )

        # Get unique labels for consistent ordering
        unique_labels = sorted(results["test_labels"].unique())

        # Decision Tree confusion matrix
        cm_tree = confusion_matrix(
            results["test_labels"], results["predictions"]["tree"], labels=unique_labels
        )
        sns.heatmap(
            cm_tree,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax1,
            xticklabels=unique_labels,
            yticklabels=unique_labels,
        )
        ax1.set_title("Decision Tree Confusion Matrix", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Predicted Label", fontsize=12)
        ax1.set_ylabel("Actual Label", fontsize=12)

        # Logistic Regression confusion matrix
        cm_logreg = confusion_matrix(
            results["test_labels"],
            results["predictions"]["logreg"],
            labels=unique_labels,
        )
        sns.heatmap(
            cm_logreg,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax2,
            xticklabels=unique_labels,
            yticklabels=unique_labels,
        )
        ax2.set_title(
            "Logistic Regression Confusion Matrix", fontsize=14, fontweight="bold"
        )
        ax2.set_xlabel("Predicted Label", fontsize=12)
        ax2.set_ylabel("Actual Label", fontsize=12)

        plt.tight_layout()

        # Save plot if configured
        if CONFIG["visualization"]["save_plots"]:
            output_path = (
                CONFIG["paths"]["outputs_dir"]
                / CONFIG["output_files"]["confusion_matrices"]
            )
            plt.savefig(
                output_path, dpi=CONFIG["visualization"]["dpi"], bbox_inches="tight"
            )
            logger.info(f"Confusion matrices plot saved to: {output_path}")

        plt.show()

    def save_models(self, output_dir: Optional[str] = None):
        """
        Save trained models and preprocessing objects.

        Args:
            output_dir: Directory to save models (uses config default if None)
        """
        if output_dir is None:
            output_path = CONFIG["paths"]["models_dir"]
        else:
            output_path = Path(output_dir)

        output_path.mkdir(exist_ok=True)

        logger.info(f"Saving models to {output_path}")

        # Save models using configured file names
        model_files = CONFIG["model_files"]
        for name, model in self.models.items():
            if name in model_files:
                file_path = output_path / model_files[name]
                joblib.dump(model, file_path)
                logger.info(f"Saved {name} model to {file_path}")

        # Save preprocessing objects
        if self.scaler is not None:
            scaler_path = output_path / model_files["scaler"]
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Saved scaler to {scaler_path}")

        if self.pca is not None:
            pca_path = output_path / model_files["pca"]
            joblib.dump(self.pca, pca_path)
            logger.info(f"Saved PCA transformer to {pca_path}")

        logger.info("All models saved successfully")

    def load_models(self, model_dir: Optional[str] = None):
        """
        Load trained models and preprocessing objects.

        Args:
            model_dir: Directory containing saved models (uses config default if None)

        Raises:
            FileNotFoundError: If model directory or files are not found
        """
        if model_dir is None:
            model_path = CONFIG["paths"]["models_dir"]
        else:
            model_path = Path(model_dir)

        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        logger.info(f"Loading models from {model_path}")

        # Load models using configured file names
        model_files = CONFIG["model_files"]
        self.models = {}

        for model_name, file_name in model_files.items():
            if model_name not in ["scaler", "pca"]:
                file_path = model_path / file_name
                if file_path.exists():
                    self.models[model_name] = joblib.load(file_path)
                    logger.info(f"Loaded {model_name} model from {file_path}")
                else:
                    logger.warning(f"Model file not found: {file_path}")

        # Load preprocessing objects
        scaler_file = model_path / model_files["scaler"]
        if scaler_file.exists():
            self.scaler = joblib.load(scaler_file)
            logger.info(f"Loaded scaler from {scaler_file}")
        else:
            logger.warning(f"Scaler file not found: {scaler_file}")

        pca_file = model_path / model_files["pca"]
        if pca_file.exists():
            self.pca = joblib.load(pca_file)
            logger.info(f"Loaded PCA transformer from {pca_file}")
        else:
            logger.warning(f"PCA file not found: {pca_file}")

        if not self.models:
            raise FileNotFoundError("No model files found in the specified directory")

        logger.info("Models loaded successfully")

    def predict(
        self, features: np.ndarray, model_name: str = "logistic_regression"
    ) -> np.ndarray:
        """
        Make predictions using a trained model.

        Args:
            features: Feature matrix
            model_name: Name of the model to use for prediction

        Returns:
            Predictions array
        """
        if model_name not in self.models:
            raise ValueError(
                f"Model '{model_name}' not found. Available models: {list(self.models.keys())}"
            )

        # Preprocess features
        scaled_features = self.scaler.transform(features)

        # Apply PCA if available
        if self.pca is not None:
            features_processed = self.pca.transform(scaled_features)
        else:
            features_processed = scaled_features

        # Make predictions
        predictions = self.models[model_name].predict(features_processed)

        return predictions

    def run_full_pipeline(
        self, balance_data: bool = True, n_components: int = 6
    ) -> Dict[str, Any]:
        """
        Run the complete machine learning pipeline.

        Args:
            balance_data: Whether to balance the dataset
            n_components: Number of PCA components

        Returns:
            Dictionary containing all results
        """
        logger.info("Starting full pipeline...")

        # Load data
        data = self.load_data()

        # Analyze correlations
        corr_matrix = self.analyze_correlations(
            data.drop(columns=["genre_top", "track_id"])
        )

        # Balance data if requested
        if balance_data:
            data = self.balance_dataset(data)

        # Preprocess data
        scaled_features, labels = self.preprocess_data(data)

        # Perform PCA
        pca_features = self.perform_pca(scaled_features, n_components)

        # Train models
        training_results = self.train_models(pca_features, labels)

        # Cross-validation
        cv_results = self.cross_validate_models(pca_features, labels)

        # Plot confusion matrices
        self.plot_confusion_matrices(training_results)

        # Save models
        self.save_models()

        # Compile results
        results = {
            "correlation_matrix": corr_matrix,
            "training_results": training_results,
            "cv_results": cv_results,
            "data_info": {
                "original_size": len(self.load_data()),
                "final_size": len(data),
                "features": pca_features.shape[1],
                "genres": labels.unique().tolist(),
            },
        }

        logger.info("Pipeline completed successfully!")
        return results


def main():
    """Main function to run the song genre classification system."""
    try:
        logger.info("Starting Song Genre Classification System")

        # Get configuration
        model_config = get_model_config()

        # Initialize classifier
        classifier = SongGenreClassifier(
            data_dir=str(CONFIG["paths"]["data_dir"]),
            random_state=model_config["random_state"],
        )

        # Run full pipeline
        results = classifier.run_full_pipeline(
            balance_data=model_config["balance_dataset"],
            n_components=model_config["pca_components"],
        )

        # Print summary
        print("\n" + "=" * 60)
        print("SONG GENRE CLASSIFICATION RESULTS")
        print("=" * 60)
        print(f"Dataset size: {results['data_info']['final_size']} samples")
        print(f"Features used: {results['data_info']['features']} (after PCA)")
        print(f"Genres: {', '.join(results['data_info']['genres'])}")
        print(f"Environment: {CONFIG['environment']}")
        print("\nCross-validation Results:")
        print(
            f"Decision Tree: {results['cv_results']['decision_tree_mean']:.3f} ± {results['cv_results']['decision_tree_std']:.3f}"
        )
        print(
            f"Logistic Regression: {results['cv_results']['logistic_regression_mean']:.3f} ± {results['cv_results']['logistic_regression_std']:.3f}"
        )
        print("\nOutput Files:")
        print(f"Models saved to: {CONFIG['paths']['models_dir']}")
        print(f"Plots saved to: {CONFIG['paths']['outputs_dir']}")
        print(f"Logs saved to: {CONFIG['paths']['logs_dir']}")
        print("=" * 60)

        logger.info("Song Genre Classification System completed successfully")

    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        logger.error("Please ensure data files are in the correct location")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Data validation error: {str(e)}")
        logger.error("Please check your data format and quality")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error in main pipeline: {str(e)}")
        logger.error("Please check the logs for more details")
        sys.exit(1)


if __name__ == "__main__":
    main()
