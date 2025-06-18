#!/bin/bash

echo "========================================"
echo "    AI Resume Screening System"
echo "========================================"
echo

echo "Testing dependencies..."
python3 test_installation.py

if [ $? -eq 0 ]; then
    echo
    echo "Starting the application..."
    echo "The app will open in your default browser."
    echo "If it doesn't open automatically, go to: http://localhost:8501"
    echo
    echo "Press Ctrl+C to stop the application."
    echo
    streamlit run resume_screener.py
else
    echo
    echo "Please install dependencies first:"
    echo "pip install -r requirements.txt"
    echo
    read -p "Press Enter to continue..."
fi 