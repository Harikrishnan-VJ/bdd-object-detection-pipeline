# Instructions for Using the Installation Script

Step-by-step instructions on how to use the `install.sh` script to set up a Python virtual environment and install project dependencies from `requirements.txt`.

## Prerequisites
- Python 3 installed on your system (the script defaults to `python3`).
- `requirements.txt` exists in the root of your repository. If it's elsewhere, configure the path accordingly.

## How to Use the Script

1. **Make the Script Executable**:
   - Open a terminal in repository's root directory.
   - Run the following command:
```
    chmod +x install.sh
```

2. **Configure Variables (Optional)**:
- The script uses environment variables for configuration. Set them before running the script to override defaults.
- Available variables:
- `VENV_NAME`: Name of the virtual environment directory (default: `venv`).
  - Example: `VENV_NAME=myenv` to create a directory named `myenv`.
- `PYTHON_BIN`: Python binary to use (default: `python3`).
  - Example: `PYTHON_BIN=python3` if your system uses Python 3.
- `REQUIREMENTS_FILE`: Path to the requirements file (default: `requirements.txt`).
  - Example: `REQUIREMENTS_FILE=dev-requirements.txt` for a different file.


3. **Run the Script**:
- In the terminal, execute:
```
    ./install.sh
```         
The script will:
- Create the virtual environment.
- Activate it.
- Install the packages listed in the requirements file.
- Deactivate the environment.

4. **Activate the Virtual Environment Manually**:
- After installation, to start using the environment:
```
source venv/bin/activate  # Replace 'venv' with the VENV_NAME 
```
5. **To deactivate**:
```
deactivate
```


