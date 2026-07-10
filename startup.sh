#!/bin/bash

# Install Python dependencies if not already installed
if [ ! -d "/home/site/wwwroot/antenv" ]; then
    echo "Virtual environment not found. Creating and installing dependencies..."
    python -m venv /home/site/wwwroot/antenv
    /home/site/wwwroot/antenv/bin/pip install -r /home/site/wwwroot/requirements.txt
else
    echo "Virtual environment found. Skipping install."
fi

# Activate the virtual environment
source /home/site/wwwroot/antenv/bin/activate

# Build frontend if not already built
if [ ! -d "/home/site/wwwroot/frontend/dist" ]; then
    echo "Frontend build not found. Building..."
    cd /home/site/wwwroot/frontend && npm install && npm run build && cd /home/site/wwwroot
else
    echo "Frontend build found. Skipping."
fi

# Start the FastAPI server
cd /home/site/wwwroot
python -m uvicorn main:app --host 0.0.0.0 --port 8000
