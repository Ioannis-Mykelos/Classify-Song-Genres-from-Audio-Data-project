# Song Genre Classification System

A professional machine learning system for classifying songs into Rock and Hip-Hop genres using audio features from The Echo Nest dataset.

## 🎵 Overview

This project implements a complete machine learning pipeline for music genre classification, featuring:

- **Data preprocessing** with feature scaling and PCA dimensionality reduction
- **Multiple ML algorithms** including Decision Trees and Logistic Regression
- **Model evaluation** with cross-validation and comprehensive metrics
- **Production-ready code** with proper error handling, logging, and configuration management
- **Command-line interface** for easy usage
- **Comprehensive testing** with unit tests
- **Professional documentation** and code organization

## 🚀 Features

### Core Functionality
- Load and merge CSV and JSON datasets
- Feature correlation analysis with visualizations
- Principal Component Analysis (PCA) for dimensionality reduction
- Dataset balancing to handle class imbalance
- Model training with Decision Trees and Logistic Regression
- Cross-validation for robust model evaluation
- Model persistence (save/load trained models)
- Prediction on new data

### Production Features
- Comprehensive error handling and logging
- Configuration management system
- Command-line interface
- Unit test suite
- Professional code structure
- Type hints and documentation
- Data validation
- Visualization with proper styling

## 📁 Project Structure

```
Classify-Song-Genres-from-Audio-Data-project/
├── song_genre_classifier.py    # Main classifier class
├── config.py                   # Configuration management
├── cli.py                      # Command-line interface
├── test_song_classifier.py     # Unit tests
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── data/                       # Data directory
│   ├── fma-rock-vs-hiphop.csv
│   └── echonest-metrics.json
├── models/                     # Saved models (created after training)
├── logs/                       # Log files
└── outputs/                    # Generated plots and results
```

## 🛠️ Installation

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Classify-Song-Genres-from-Audio-Data-project
   ```

2. **Set up development environment (recommended):**
   ```bash
   python scripts/setup_dev.py
   ```

3. **Or manual setup:**
   ```bash
   # Create virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt

   # Install pre-commit hooks
   pre-commit install
   ```

4. **Prepare your data:**
   - Place your `fma-rock-vs-hiphop.csv` and `echonest-metrics.json` files in the `data/` directory
   - Ensure the data files have the expected structure (see Data Format section)

### Development Setup

For contributors and developers:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
pre-commit install --hook-type commit-msg

# Run all quality checks
pre-commit run --all-files

# Run tests
pytest test_song_classifier.py -v --cov

# Run linting
pylint song_genre_classifier.py config.py cli.py
```

## 📊 Data Format

### Input Files

**fma-rock-vs-hiphop.csv:**
```csv
track_id,genre_top
1,Rock
2,Hip-Hop
3,Rock
...
```

**echonest-metrics.json:**
```json
[
  {
    "track_id": 1,
    "danceability": 0.5,
    "energy": 0.7,
    "speechiness": 0.1,
    "acousticness": 0.2,
    "instrumentalness": 0.0,
    "liveness": 0.1,
    "valence": 0.6
  },
  ...
]
```

## 🎯 Usage

### Command Line Interface

#### 1. Train Models
```bash
# Basic training
python cli.py train

# Advanced training with custom parameters
python cli.py train --data-dir data --balance --pca-components 6 --random-state 42
```

#### 2. Make Predictions
```bash
# Predict on new data
python cli.py predict --input new_songs.csv --output predictions.csv --model logistic_regression
```

#### 3. Evaluate Models
```bash
# Evaluate model performance
python cli.py evaluate --test-data test_data.csv --model decision_tree
```

### Python API

```python
from song_genre_classifier import SongGenreClassifier

# Initialize classifier
classifier = SongGenreClassifier(data_dir="data")

# Train models
results = classifier.run_full_pipeline(balance_data=True, n_components=6)

# Make predictions
predictions = classifier.predict(new_features, model_name="logistic_regression")

# Save/load models
classifier.save_models("models")
classifier.load_models("models")
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python -m pytest test_song_classifier.py -v

# Run with coverage
python -m pytest test_song_classifier.py --cov=song_genre_classifier --cov-report=html

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Skip slow tests
```

## 🔄 Continuous Integration

This project uses GitHub Actions for automated testing and code quality checks:

### Automated Checks

Every push and pull request triggers:

- **Pre-commit hooks**: Code formatting, linting, and quality checks
- **Pylint**: Python code analysis and style checking
- **Pytest**: Unit and integration tests with coverage reporting
- **Security scanning**: Bandit security analysis
- **Dependency checking**: Vulnerability scanning with pip-audit
- **Type checking**: MyPy static type analysis

### Quality Gates

- Code coverage must be ≥80%
- All tests must pass
- No security vulnerabilities
- Code must pass all linting checks
- Pre-commit hooks must pass

### View CI Status

Check the [Actions tab](https://github.com/yourusername/Classify-Song-Genres-from-Audio-Data-project/actions) in your repository to see the latest CI runs.

## 📈 Model Performance

The system achieves the following performance metrics:

- **Decision Tree**: ~87% accuracy with cross-validation
- **Logistic Regression**: ~87% accuracy with cross-validation
- **Balanced Dataset**: Improved performance on minority class (Hip-Hop)
- **Feature Reduction**: PCA reduces dimensionality while maintaining performance

## 🔧 Configuration

The system uses a centralized configuration system (`config.py`) that allows you to customize:

- Data file paths
- Model parameters
- Feature processing settings
- Visualization options
- Logging configuration
- Environment-specific settings

### Key Configuration Options

```python
# Model settings
MODEL_CONFIG = {
    'random_state': 10,
    'test_size': 0.2,
    'cv_folds': 10,
    'pca_components': 6,
    'balance_dataset': True
}

# Feature processing
FEATURE_CONFIG = {
    'target_column': 'genre_top',
    'scale_features': True,
    'apply_pca': True
}
```

## 📊 Outputs

The system generates several outputs:

### Visualizations
- `correlation_matrix.png` - Feature correlation heatmap
- `pca_analysis.png` - PCA scree plot and cumulative variance
- `confusion_matrices.png` - Model confusion matrices

### Models
- `decision_tree.pkl` - Trained decision tree model
- `logistic_regression.pkl` - Trained logistic regression model
- `scaler.pkl` - Feature scaler
- `pca.pkl` - PCA transformer

### Logs
- `song_classifier.log` - Application logs

## 🚀 Production Deployment

### Environment Variables
```bash
export ENVIRONMENT=production
export DATA_DIR=/path/to/data
export MODEL_DIR=/path/to/models
```

### Docker Support
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "cli.py", "train"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Workflow

```bash
# 1. Make changes to your code
# 2. Run pre-commit hooks manually (optional)
pre-commit run --all-files

# 3. Run tests
pytest test_song_classifier.py -v --cov

# 4. Run specific quality checks
black song_genre_classifier.py config.py cli.py    # Format code
isort song_genre_classifier.py config.py cli.py    # Sort imports
pylint song_genre_classifier.py config.py cli.py   # Lint code
mypy song_genre_classifier.py config.py cli.py     # Type checking
bandit -r song_genre_classifier.py config.py cli.py # Security scan

# 5. Commit your changes (pre-commit hooks run automatically)
git add .
git commit -m "feat: add new feature"

# 6. Push to trigger CI/CD pipeline
git push origin feature-branch
```

### Code Quality Tools

- **Black**: Code formatting
- **isort**: Import sorting
- **Flake8**: Linting and style checking
- **Pylint**: Advanced code analysis
- **MyPy**: Static type checking
- **Bandit**: Security vulnerability scanning
- **Pytest**: Testing framework with coverage
- **Pre-commit**: Git hooks for quality assurance

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- The Echo Nest for providing the audio features dataset
- FMA (Free Music Archive) for the genre labels
- Scikit-learn for the machine learning algorithms
- The open-source community for the various Python libraries used

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/your-repo/issues) page
2. Create a new issue with detailed information
3. Include error messages and system information

## 🔮 Future Enhancements

- [ ] Support for more genres (Pop, Jazz, Classical, etc.)
- [ ] Deep learning models (CNN, RNN)
- [ ] Real-time audio classification
- [ ] Web API interface
- [ ] Model versioning and A/B testing
- [ ] Automated hyperparameter tuning
- [ ] Integration with music streaming APIs

---

**Happy classifying! 🎵🤖**