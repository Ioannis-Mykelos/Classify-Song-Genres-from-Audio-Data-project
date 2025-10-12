#!/usr/bin/env python3
"""
Development environment setup script for Song Genre Classification Project.

This script helps set up the development environment with all necessary
dependencies and pre-commit hooks.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up Song Genre Classification Development Environment")
    print("=" * 60)

    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)

    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

    # Create virtual environment if it doesn't exist
    venv_path = Path(".venv")
    if not venv_path.exists():
        print("\n🔄 Creating virtual environment...")
        if not run_command("python -m venv .venv", "Creating virtual environment"):
            sys.exit(1)

    # Determine command paths based on OS
    if os.name == "nt":  # Windows
        pip_cmd = ".venv\\Scripts\\pip"
        python_cmd = ".venv\\Scripts\\python"
    else:  # Unix/Linux/macOS
        pip_cmd = ".venv/bin/pip"
        python_cmd = ".venv/bin/python"

    # Upgrade pip
    if not run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip"):
        sys.exit(1)

    # Install dependencies
    if not run_command(
        f"{pip_cmd} install -r requirements.txt", "Installing dependencies"
    ):
        sys.exit(1)

    # Install pre-commit hooks
    if not run_command(
        f"{python_cmd} -m pre_commit install", "Installing pre-commit hooks"
    ):
        sys.exit(1)

    # Install commit-msg hook
    if not run_command(
        f"{python_cmd} -m pre_commit install --hook-type commit-msg",
        "Installing commit-msg hook",
    ):
        sys.exit(1)

    # Run pre-commit on all files
    if not run_command(
        f"{python_cmd} -m pre_commit run --all-files", "Running pre-commit on all files"
    ):
        print("⚠️  Pre-commit found some issues. Please fix them before committing.")

    # Run tests
    if not run_command(
        f"{python_cmd} -m pytest test_song_classifier.py -v", "Running tests"
    ):
        print("⚠️  Some tests failed. Please check the test output.")

    print("\n" + "=" * 60)
    print("🎉 Development environment setup completed!")
    print("\n📋 Next steps:")
    print("1. Activate the virtual environment:")
    if os.name == "nt":
        print("   .venv\\Scripts\\activate")
    else:
        print("   source .venv/bin/activate")
    print("2. Make your changes to the code")
    print("3. Run tests: pytest test_song_classifier.py -v")
    print("4. Run linting: pylint song_genre_classifier.py config.py cli.py")
    print("5. Commit your changes (pre-commit hooks will run automatically)")
    print("\n🔗 Useful commands:")
    print("- Format code: black song_genre_classifier.py config.py cli.py")
    print("- Sort imports: isort song_genre_classifier.py config.py cli.py")
    print("- Type checking: mypy song_genre_classifier.py config.py cli.py")
    print("- Security scan: bandit -r song_genre_classifier.py config.py cli.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
