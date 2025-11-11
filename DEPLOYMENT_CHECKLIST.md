# 🚀 Natron Transformer - Deployment Checklist

## Pre-Deployment Verification

### ✅ System Requirements
- [ ] Ubuntu 20.04+ / Debian 11+
- [ ] Python 3.10+
- [ ] NVIDIA GPU with 6GB+ VRAM
- [ ] CUDA 11.8+
- [ ] 16GB+ RAM
- [ ] 10GB+ free storage

### ✅ Installation
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Expected: CUDA: True
```

### ✅ Data Preparation
```bash
# Option A: Generate sample data (testing)
python scripts/generate_sample_data.py --candles 10000 --output data_export.csv

# Option B: Use real market data
# Ensure CSV has columns: time, open, high, low, close, volume
```

### ✅ Training
```bash
# Start training pipeline
python train.py --data data_export.csv --config config/config.yaml

# Monitor progress
watch -n 1 nvidia-smi  # GPU usage
tail -f logs/training.log  # Training logs
```

### ✅ Model Validation
After training, verify these files exist:
- [ ] `model/natron_v2.pt` - Final trained model
- [ ] `model/natron_v2_best.pt` - Best validation model
- [ ] `model/scaler.pkl` - Feature scaler
- [ ] `model/pretrained_encoder.pt` - Pretrained encoder (optional)

### ✅ Model Evaluation
```bash
python scripts/evaluate_model.py \
    --model model/natron_v2.pt \
    --data data_export.csv \
    --output-dir evaluation

# Check results in evaluation/ directory
```

### ✅ API Server Testing
```bash
# 1. Start server
python src/inference/api_server.py \
    --model model/natron_v2.pt \
    --config config/config.yaml \
    --scaler model/scaler.pkl

# 2. Test health endpoint
curl http://localhost:5000/health

# Expected: {"status": "healthy", "model_loaded": true}

# 3. Test prediction (create test_request.json first)
curl -X POST http://localhost:5000/predict \
    -H 'Content-Type: application/json' \
    -d @test_request.json
```

### ✅ Performance Benchmarks
Run these tests:
- [ ] Inference latency < 50ms
- [ ] GPU memory usage < 4GB
- [ ] API response time < 100ms
- [ ] Throughput > 20 req/sec

### ✅ Production Deployment

#### Docker Deployment
```bash
# 1. Create Dockerfile
cat > Dockerfile <<'DOCKER'
FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "src/inference/api_server.py"]
DOCKER

# 2. Build image
docker build -t natron-transformer:latest .

# 3. Run container
docker run -d -p 5000:5000 --gpus all natron-transformer:latest
```

#### Systemd Service
```bash
# 1. Create service file
sudo nano /etc/systemd/system/natron.service

# 2. Add configuration:
[Unit]
Description=Natron Transformer API
After=network.target

[Service]
Type=simple
User=natron
WorkingDirectory=/opt/natron
ExecStart=/opt/natron/venv/bin/python src/inference/api_server.py
Restart=always

[Install]
WantedBy=multi-user.target

# 3. Enable and start
sudo systemctl enable natron
sudo systemctl start natron
```

#### Load Balancer (Optional)
```bash
# Install nginx
sudo apt install nginx

# Configure reverse proxy
sudo nano /etc/nginx/sites-available/natron

# Add:
upstream natron_backend {
    server 127.0.0.1:5000;
    server 127.0.0.1:5001;
    server 127.0.0.1:5002;
}

server {
    listen 80;
    location / {
        proxy_pass http://natron_backend;
    }
}
```

### ✅ Monitoring Setup
```bash
# 1. GPU monitoring
watch -n 1 nvidia-smi

# 2. Server logs
tail -f logs/api_server.log

# 3. Performance metrics
# Install Prometheus + Grafana (optional)
```

### ✅ MQL5 Integration

Create Expert Advisor:
```mql5
//+------------------------------------------------------------------+
//| Natron EA Integration                                             |
//+------------------------------------------------------------------+
input string ServerURL = "http://your-server:5000/predict";

// Collect 96 candles
string candles_json = BuildCandlesJSON();

// Send HTTP request
string response = SendHTTPRequest(ServerURL, candles_json);

// Parse response
double buy_prob = ParseBuyProb(response);
double sell_prob = ParseSellProb(response);
string regime = ParseRegime(response);

// Trading logic
if(buy_prob > 0.7 && regime == "BULL_STRONG") {
    // Place buy order
    OrderSend(Symbol(), OP_BUY, 0.1, Ask, 3, Ask-50*Point, Ask+100*Point);
}
```

### ✅ Security Checklist
- [ ] API authentication enabled
- [ ] HTTPS/SSL certificates configured
- [ ] Rate limiting implemented
- [ ] Input validation on all endpoints
- [ ] Firewall rules configured
- [ ] Secrets stored securely (not in code)

### ✅ Backup & Recovery
```bash
# Backup models
tar -czf natron_backup_$(date +%Y%m%d).tar.gz model/ config/

# Backup to remote
rsync -avz model/ user@backup-server:/backups/natron/
```

### ✅ Maintenance Schedule
- [ ] Daily: Monitor prediction accuracy
- [ ] Weekly: Check model performance metrics
- [ ] Monthly: Retrain model with new data
- [ ] Quarterly: Full system audit

### ✅ Troubleshooting

**Issue**: CUDA out of memory
**Solution**: Reduce batch_size in config.yaml

**Issue**: Slow inference
**Solution**: Enable mixed precision, upgrade GPU

**Issue**: Poor predictions
**Solution**: Retrain with more/better data

**Issue**: API timeout
**Solution**: Increase worker threads, optimize feature generation

### ✅ Go-Live Checklist
- [ ] All tests passing
- [ ] Documentation reviewed
- [ ] Backups configured
- [ ] Monitoring active
- [ ] Team trained
- [ ] Rollback plan ready
- [ ] Emergency contacts listed

---

## 🎯 Sign-Off

- [ ] Development: ✅ Complete
- [ ] Testing: ✅ Complete
- [ ] Documentation: ✅ Complete
- [ ] Deployment: ⬜ Ready
- [ ] Production: ⬜ Pending

**Date**: _______________  
**Deployed By**: _______________  
**Approved By**: _______________  

---

**Status**: ✅ READY FOR DEPLOYMENT
