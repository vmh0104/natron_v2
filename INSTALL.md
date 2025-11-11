# 🚀 Installation & Setup Guide

## Quick Installation

### 1. System Requirements

**Minimum:**
- Python 3.10+
- 8GB RAM
- CPU with AVX2 support

**Recommended:**
- Python 3.10+
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM
- CUDA 11.8+
- Ubuntu 20.04+ or similar Linux distribution

### 2. Install Python Dependencies

```bash
# Update pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

### 3. GPU Setup (Recommended)

If you have a CUDA-capable GPU:

```bash
# Check CUDA installation
nvidia-smi

# Verify PyTorch can use CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0)}')"
```

If CUDA is not available, install CUDA toolkit:
```bash
# For Ubuntu
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

### 4. Verify Installation

Run the test suite to verify everything is installed correctly:

```bash
# Test configuration
python -c "from config import load_config; c = load_config(); print('✅ Config OK')"

# Test feature engine
python -c "from feature_engine import FeatureEngine; e = FeatureEngine(); print('✅ Feature Engine OK')"

# Test model
python -c "from model import create_model; from config import load_config; m = create_model(load_config()); print('✅ Model OK')"
```

## Detailed Installation Steps

### Option 1: Using Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using Conda

```bash
# Create conda environment
conda create -n natron python=3.10

# Activate environment
conda activate natron

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

### Option 3: System-wide Installation

```bash
# Install system-wide (requires sudo)
sudo pip install -r requirements.txt
```

## Data Preparation

### Format Your Data

Your input CSV must have these columns:
- `time`: Timestamp (format: "YYYY-MM-DD HH:MM:SS")
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price
- `close`: Closing price
- `volume`: Trading volume

Example:
```csv
time,open,high,low,close,volume
2023-01-01 00:00:00,1.0500,1.0520,1.0490,1.0510,10000
2023-01-01 00:15:00,1.0510,1.0530,1.0505,1.0525,12000
2023-01-01 00:30:00,1.0525,1.0535,1.0515,1.0530,11500
...
```

### Export from MetaTrader

Use this MQL5 script to export data:

```mql5
//+------------------------------------------------------------------+
//| Export OHLCV data to CSV                                         |
//+------------------------------------------------------------------+
void ExportData()
{
    string filename = "data_export.csv";
    int handle = FileOpen(filename, FILE_WRITE|FILE_CSV, ",");
    
    if(handle != INVALID_HANDLE)
    {
        // Write header
        FileWrite(handle, "time", "open", "high", "low", "close", "volume");
        
        // Export data (e.g., last 10000 bars)
        int bars = 10000;
        for(int i = bars - 1; i >= 0; i--)
        {
            datetime time = iTime(Symbol(), Period(), i);
            double open = iOpen(Symbol(), Period(), i);
            double high = iHigh(Symbol(), Period(), i);
            double low = iLow(Symbol(), Period(), i);
            double close = iClose(Symbol(), Period(), i);
            long volume = iVolume(Symbol(), Period(), i);
            
            FileWrite(handle, 
                      TimeToString(time, TIME_DATE|TIME_MINUTES),
                      open, high, low, close, volume);
        }
        
        FileClose(handle);
        Print("Data exported to ", filename);
    }
}
```

## Configuration

### Create Your Config File

```bash
# Copy example config
cp config_example.yaml my_config.yaml

# Edit with your preferred editor
nano my_config.yaml
# or
vim my_config.yaml
```

### Key Settings to Adjust

```yaml
# Data paths
data:
  csv_path: "path/to/your/data_export.csv"
  batch_size: 128  # Reduce if you have memory issues

# Model size (reduce for smaller GPUs)
model:
  d_model: 256      # 128 for smaller GPUs
  nhead: 8          # 4 for smaller GPUs
  num_encoder_layers: 6  # 4 for smaller GPUs

# Training
train:
  epochs: 100
  learning_rate: 0.0001

# Device
device: "cuda"  # or "cpu" if no GPU
```

## First Run

### Test with Small Dataset

Before running on your full dataset, test with a small sample:

```bash
# Create a small test dataset (first 1000 rows)
head -n 1001 data_export.csv > test_data.csv

# Run training on test data
python main.py train --data test_data.csv --epochs 5 --skip-pretrain

# This should complete quickly and verify everything works
```

### Full Training Pipeline

Once verified, run the full pipeline:

```bash
# Full pipeline with all phases
python main.py train \
  --data data_export.csv \
  --epochs 100 \
  --pretrain-epochs 50 \
  --config my_config.yaml
```

## Common Issues & Solutions

### Issue: "CUDA out of memory"

**Solution:**
```yaml
# In config.yaml, reduce batch size
data:
  batch_size: 64  # or 32

# Reduce model size
model:
  d_model: 128
  num_encoder_layers: 4
```

### Issue: "ModuleNotFoundError"

**Solution:**
```bash
# Ensure all packages are installed
pip install -r requirements.txt

# Check Python path
which python
python --version
```

### Issue: "FileNotFoundError: data_export.csv"

**Solution:**
```bash
# Check file exists
ls -lh data_export.csv

# Use absolute path
python main.py train --data /full/path/to/data_export.csv
```

### Issue: Slow training on CPU

**Solution:**
- Install CUDA and use GPU (10-50x faster)
- Reduce model size
- Use smaller batch size
- Consider cloud GPU services (Google Colab, AWS, etc.)

### Issue: "Too many open files"

**Solution:**
```bash
# Increase file descriptor limit (Linux)
ulimit -n 4096

# Or reduce num_workers in config
data:
  num_workers: 2  # or 0
```

## Performance Optimization

### GPU Optimization

```yaml
# Enable mixed precision (2x faster)
mixed_precision: true

# Increase batch size if you have VRAM
data:
  batch_size: 256  # or higher

# Enable pin_memory for faster data loading
data:
  pin_memory: true
```

### CPU Optimization

```yaml
# Reduce model complexity
model:
  d_model: 128
  num_encoder_layers: 4

# Reduce batch size
data:
  batch_size: 32

# Disable pin_memory
data:
  pin_memory: false
```

## Next Steps

After installation:

1. **Prepare your data** - Export from MT5 or prepare CSV
2. **Configure** - Copy and edit config_example.yaml
3. **Train** - Run training pipeline
4. **Evaluate** - Check model performance
5. **Deploy** - Start API server for live trading

See [README.md](README.md) for detailed usage instructions.

## Support

If you encounter issues:

1. Check the [Troubleshooting](README.md#-troubleshooting) section
2. Search [GitHub Issues](https://github.com/your-repo/issues)
3. Create a new issue with:
   - Python version
   - CUDA version (if using GPU)
   - Error message
   - Steps to reproduce

---

**Installation complete! Ready to train your first model? 🚀**

```bash
python main.py train --data data_export.csv --epochs 100
```
