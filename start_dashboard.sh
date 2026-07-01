#!/bin/bash
# NETRA Web Dashboard Launcher

set -e

VENV_PYTHON="./venv/bin/python3"

echo "=========================================="
echo "🚦 Starting NETRA Web Dashboard..."
echo "=========================================="
echo ""

if [ ! -x "$VENV_PYTHON" ]; then
    echo "❌ Virtual environment not found at ./venv"
    echo "   Create it first: python3 -m venv venv"
    echo "   Then install deps: ./venv/bin/python3 -m pip install -r requirements.txt"
    exit 1
fi

# Check if streamlit is installed
if ! "$VENV_PYTHON" -c "import streamlit" >/dev/null 2>&1; then
    echo "⚠️  Streamlit not found. Installing requirements..."
    "$VENV_PYTHON" -m pip install -r requirements.txt
fi

echo "🚀 Launching dashboard..."
echo "📱 The dashboard will open in your browser automatically"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run streamlit from virtual environment
"$VENV_PYTHON" -m streamlit run src/web_dashboard.py
