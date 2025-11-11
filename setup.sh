#!/bin/bash
# Setup script for Natron Transformer V2

set -e

echo "🧠 Natron Transformer V2 - Setup Script"
echo "========================================"

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create directories
echo "Creating directories..."
mkdir -p data
mkdir -p models
mkdir -p checkpoints/pretrain
mkdir -p checkpoints/supervised
mkdir -p logs
mkdir -p mql5

# Create .gitkeep for data directory
touch data/.gitkeep

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Check CUDA availability
echo "Checking CUDA availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Test component imports
echo "Testing component imports..."
python3 -c "
from src.feature_engine import FeatureEngine
from src.label_generator import LabelGeneratorV2
from src.sequence_creator import SequenceCreator
from src.model import NatronTransformer
print('✅ All imports successful')
"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Place your OHLCV data at: data/data_export.csv"
echo "2. Configure settings in: config.yaml"
echo "3. Run training: python train.py"
echo "4. Test components: python test_components.py"
