#!/bin/bash

# Create Python venv and install requirements.

# Configure variables
VENV_NAME="bdd-venv"  # Name of the venv directory
PYTHON_BIN="python3"  # Python binary to use 
REQUIREMENTS_FILE="requirements.txt"  # Path to the requirements file

# Check if the requirements file exists
if [ ! -f $REQUIREMENTS_FILE ]; then
    echo "Error: Requirements file $REQUIREMENTS_FILE not found."
    exit 1
fi

# Create venv
echo "Creating virtual environment in '$VENV_NAME'..."
$PYTHON_BIN -m venv "$VENV_NAME"

# Activate venv
echo "Activating virtual environment..."
source "$VENV_NAME/bin/activate"

# Install requirements
echo "Installing requirements from '$REQUIREMENTS_FILE'..."
pip3 install -r "$REQUIREMENTS_FILE"

# Deactivate the venv after installation
deactivate

echo "Installation complete. To use the venv, run: source $VENV_NAME/bin/activate"