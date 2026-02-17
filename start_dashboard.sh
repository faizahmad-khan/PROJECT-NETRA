#!/bin/bash
# NETRA Web Dashboard Launcher

echo "=========================================="
echo "🚦 Starting NETRA Web Dashboard..."
echo "=========================================="
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "⚠️  Streamlit not found. Installing requirements..."
    pip install -r requirements.txt
fi

echo "🚀 Launching dashboard..."
echo "📱 The dashboard will open in your browser automatically"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run streamlit from virtual environment
./venv/bin/streamlit run src/web_dashboard.py
