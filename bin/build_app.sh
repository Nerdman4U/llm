#!/bin/bash
# NOTE: remember use "$ source build_app" to alter current shell not subshell.

# version to be build is at pyproject.toml
# - should have a python script to extract version from there
# - this version should then be installed
# - script name could be "build_and_install_current"

source .activate
echo "Virtual environment activated"

echo "Building project"
python -m build

echo "Project built successfully"