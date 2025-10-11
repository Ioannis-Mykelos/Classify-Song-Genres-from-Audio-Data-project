"""
Setup script for the Song Genre Classification System.

This script helps set up the project environment and install dependencies.
"""

from pathlib import Path

from setuptools import find_packages, setup

# Read README file
readme_path = Path(__file__).parent / "README.md"
long_description = (
    readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
)

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = requirements_path.read_text().strip().split("\n")
    requirements = [
        req.strip() for req in requirements if req.strip() and not req.startswith("#")
    ]

setup(
    name="song-genre-classifier",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A professional machine learning system for classifying songs into Rock and Hip-Hop genres",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Classify-Song-Genres-from-Audio-Data-project",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "song-classifier=cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
