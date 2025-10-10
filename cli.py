#!/usr/bin/env python3
"""
Command-line interface for the Song Genre Classification system.

This module provides a command-line interface for training models,
making predictions, and managing the song classification system.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

from song_genre_classifier import SongGenreClassifier
from config import get_config, get_data_paths


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def train_command(args):
    """Handle the train command."""
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize classifier
        classifier = SongGenreClassifier(
            data_dir=args.data_dir,
            random_state=args.random_state
        )
        
        logger.info("Starting model training...")
        
        # Run training pipeline
        results = classifier.run_full_pipeline(
            balance_data=args.balance,
            n_components=args.pca_components
        )
        
        # Print results
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Dataset size: {results['data_info']['final_size']} samples")
        print(f"Features used: {results['data_info']['features']} (after PCA)")
        print(f"Genres: {', '.join(results['data_info']['genres'])}")
        print("\nCross-validation Results:")
        print(f"Decision Tree: {results['cv_results']['decision_tree_mean']:.3f} ± {results['cv_results']['decision_tree_std']:.3f}")
        print(f"Logistic Regression: {results['cv_results']['logistic_regression_mean']:.3f} ± {results['cv_results']['logistic_regression_std']:.3f}")
        print("="*60)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)


def predict_command(args):
    """Handle the predict command."""
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize classifier
        classifier = SongGenreClassifier(data_dir=args.data_dir)
        
        # Load trained models
        classifier.load_models(args.model_dir)
        
        # Load input data
        import pandas as pd
        if args.input.endswith('.csv'):
            input_data = pd.read_csv(args.input)
        elif args.input.endswith('.json'):
            input_data = pd.read_json(args.input)
        else:
            raise ValueError("Input file must be CSV or JSON format")
        
        # Extract features (assuming same structure as training data)
        feature_columns = [col for col in input_data.columns 
                          if col not in ['track_id', 'genre_top']]
        features = input_data[feature_columns].values
        
        # Make predictions
        predictions = classifier.predict(features, model_name=args.model)
        
        # Create output DataFrame
        output_data = input_data.copy()
        output_data['predicted_genre'] = predictions
        
        # Save results
        output_data.to_csv(args.output, index=False)
        
        print(f"Predictions saved to: {args.output}")
        print(f"Predicted genres: {predictions}")
        
        logger.info("Prediction completed successfully!")
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        sys.exit(1)


def evaluate_command(args):
    """Handle the evaluate command."""
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize classifier
        classifier = SongGenreClassifier(data_dir=args.data_dir)
        
        # Load trained models
        classifier.load_models(args.model_dir)
        
        # Load test data
        import pandas as pd
        test_data = pd.read_csv(args.test_data)
        
        # Extract features and labels
        feature_columns = [col for col in test_data.columns 
                          if col not in ['track_id', 'genre_top']]
        features = test_data[feature_columns].values
        true_labels = test_data['genre_top'].values
        
        # Make predictions
        predictions = classifier.predict(features, model_name=args.model)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        accuracy = accuracy_score(true_labels, predictions)
        
        print(f"\nModel: {args.model}")
        print(f"Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(true_labels, predictions))
        print("\nConfusion Matrix:")
        print(confusion_matrix(true_labels, predictions))
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Song Genre Classification System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train models
  python cli.py train --data-dir data --balance --pca-components 6
  
  # Make predictions
  python cli.py predict --input test_data.csv --output predictions.csv --model logistic_regression
  
  # Evaluate model
  python cli.py evaluate --test-data test_data.csv --model decision_tree
        """
    )
    
    # Global arguments
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train classification models')
    train_parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory containing training data (default: data)'
    )
    train_parser.add_argument(
        '--balance',
        action='store_true',
        default=True,
        help='Balance the dataset (default: True)'
    )
    train_parser.add_argument(
        '--pca-components',
        type=int,
        default=6,
        help='Number of PCA components (default: 6)'
    )
    train_parser.add_argument(
        '--random-state',
        type=int,
        default=10,
        help='Random state for reproducibility (default: 10)'
    )
    train_parser.set_defaults(func=train_command)
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions on new data')
    predict_parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input file (CSV or JSON)'
    )
    predict_parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output file for predictions'
    )
    predict_parser.add_argument(
        '--model',
        type=str,
        default='logistic_regression',
        choices=['decision_tree', 'logistic_regression'],
        help='Model to use for prediction (default: logistic_regression)'
    )
    predict_parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Directory containing trained models (default: models)'
    )
    predict_parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory containing data files (default: data)'
    )
    predict_parser.set_defaults(func=predict_command)
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model performance')
    eval_parser.add_argument(
        '--test-data',
        type=str,
        required=True,
        help='Test data file (CSV)'
    )
    eval_parser.add_argument(
        '--model',
        type=str,
        default='logistic_regression',
        choices=['decision_tree', 'logistic_regression'],
        help='Model to evaluate (default: logistic_regression)'
    )
    eval_parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Directory containing trained models (default: models)'
    )
    eval_parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory containing data files (default: data)'
    )
    eval_parser.set_defaults(func=evaluate_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Check if command was provided
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()
