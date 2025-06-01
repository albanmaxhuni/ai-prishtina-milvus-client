#!/bin/bash

# Exit on error
set -e

echo "Setting up development environment..."

# Create and activate virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install development dependencies
echo "Installing development dependencies..."
pip install .[dev]

echo "Development environment setup complete!"
echo "Virtual environment is activated. Use 'deactivate' to exit the virtual environment." 