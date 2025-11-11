#!/bin/bash
# Quick demo script for Natron Transformer

set -e

echo "🧠 Natron Transformer - Quick Demo"
echo "===================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Run quick training (skip pretrain and RL for demo)
echo ""
echo "Running quick training demo (supervised only)..."
echo "This will take 5-15 minutes depending on your hardware."
echo ""

python main.py --skip-pretrain --skip-rl

echo ""
echo "✅ Training complete!"
echo ""
echo "Starting API server in background..."
python src/api_server.py &
API_PID=$!

# Wait for server to start
sleep 3

echo ""
echo "Testing API endpoint..."
curl -s -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d @test_request.json | python -m json.tool

echo ""
echo ""
echo "🎉 Demo complete!"
echo ""
echo "The API server is running on http://localhost:5000"
echo "To stop the server, run: kill $API_PID"
echo ""
echo "Try making predictions:"
echo "  curl -X POST http://localhost:5000/predict -H 'Content-Type: application/json' -d @test_request.json"
echo ""
