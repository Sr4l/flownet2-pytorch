#!/bin/bash
# FlowNet2-Pytorch installation script
# Uses uv for Python environment management with Python 3.10

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Creating Python 3.10 environment with uv..."
uv venv --python 3.10

echo "Activating environment..."
source .venv/bin/activate

echo "Installing dependencies..."
uv pip install -e . --no-build-isolation

echo "Installation complete!"
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To run FlowNet2:"
echo "  python main.py --help"