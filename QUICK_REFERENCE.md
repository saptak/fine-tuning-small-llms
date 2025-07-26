# Quick Reference Guide

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/saptak/fine-tuning-small-llms.git
cd fine-tuning-small-llms

# Quick start (automated setup)
./scripts/quick_start.sh

# Manual setup
cp .env.example .env
# Edit .env with your configuration
docker-compose up -d
```

## 📁 Repository Structure

```
fine-tuning-small-llms/
├── part1-setup/              # Environment setup and system checks
├── part2-data-preparation/    # Dataset creation and validation
├── part3-training/           # Model fine-tuning with Unsloth
├── part4-evaluation/         # Evaluation and testing frameworks
├── part5-deployment/         # Production deployment with Ollama
├── part6-production/         # Monitoring, scaling, and optimization
├── docker/                   # Docker configurations
├── scripts/                  # Utility scripts
└── docs/                     # Additional documentation
```

## 🔧 Common Commands

### Environment Setup
```bash
# Check system requirements
python part1-setup/src/system_check.py

# Setup development environment
./part1-setup/scripts/setup_environment.sh
```

### Data Preparation
```bash
# Create training dataset
python part2-data-preparation/src/dataset_creation.py \
  --output-dir ./data/datasets \
  --format alpaca

# Validate dataset quality
python part2-data-preparation/src/data_validation.py \
  --dataset ./data/datasets/sql_dataset_alpaca.json
```

### Model Training
```bash
# Train with Unsloth
python part3-training/src/fine_tune_model.py \
  --config part3-training/configs/sql_expert.yaml

# Monitor training (Jupyter)
jupyter notebook part3-training/notebooks/training_monitor.ipynb
```

### Evaluation
```bash
# Run comprehensive evaluation
python part4-evaluation/src/run_evaluation.py \
  --model-path ./models/sql-expert-merged

# Start evaluation dashboard
streamlit run part4-evaluation/src/evaluation_dashboard.py
```

### Deployment
```bash
# Deploy complete stack
docker-compose up -d

# Deploy with production config
./part5-deployment/scripts/deploy.sh

# Convert model to GGUF for Ollama
python part5-deployment/src/convert_to_gguf.py \
  --model-path ./models/sql-expert-lora
```

### Production Monitoring
```bash
# Setup monitoring
./part6-production/scripts/setup_monitoring.sh

# Monitor performance
python part6-production/src/performance_monitor.py

# Run cost analysis
python part6-production/src/cost_optimizer.py --analyze
```

## 🌐 Service URLs

| Service | URL | Default Credentials |
|---------|-----|-------------------|
| API Documentation | http://localhost:8000/docs | Bearer token: `demo-token-12345` |
| Web Interface | http://localhost:8501 | None |
| Grafana | http://localhost:3000 | admin / admin123 |
| Prometheus | http://localhost:9090 | None |
| Jupyter | http://localhost:8888 | Token in logs |

## 📊 Key Metrics

### Training Metrics
- **Loss**: Training and validation loss curves
- **Learning Rate**: Scheduler progress
- **Memory Usage**: GPU memory optimization
- **Speed**: Tokens per second, time per epoch

### Production Metrics
- **Response Time**: P50, P95, P99 latencies
- **Throughput**: Requests per second
- **Error Rate**: HTTP 4xx/5xx responses
- **Cache Hit Rate**: Response caching efficiency
- **Resource Usage**: CPU, memory, GPU utilization

## 🔑 Environment Variables

### Core Configuration
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_TOKEN=your-secure-token

# Model Configuration
MODEL_NAME=sql-expert
MAX_TOKENS=256
TEMPERATURE=0.7

# Ollama Configuration
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
```

### Training Configuration
```bash
# Weights & Biases
WANDB_PROJECT=llm-fine-tuning
WANDB_ENTITY=your-username
WANDB_API_KEY=your-api-key

# Training Parameters
BATCH_SIZE=2
LEARNING_RATE=2e-4
EPOCHS=3
```

### Production Configuration
```bash
# Monitoring
GRAFANA_ADMIN_PASSWORD=secure-password
PROMETHEUS_URL=http://prometheus:9090

# Security
JWT_SECRET_KEY=your-secret-key
ENCRYPTION_KEY=your-encryption-key

# Backup
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
S3_BUCKET=your-backup-bucket
```

## 🐛 Troubleshooting

### Common Issues

**Docker services not starting:**
```bash
# Check Docker status
docker-compose ps

# View logs
docker-compose logs -f

# Restart services
docker-compose restart
```

**CUDA/GPU not detected:**
```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

**Training out of memory:**
```bash
# Reduce batch size in config
batch_size: 1
gradient_accumulation_steps: 8

# Use smaller model
model:
  name: "unsloth/Phi-3-mini-4k-instruct-bnb-4bit"
```

**API authentication errors:**
```bash
# Check API token
curl -H "Authorization: Bearer demo-token-12345" \
  http://localhost:8000/api/v1/health
```

### Performance Optimization

**Slow inference:**
- Enable model quantization (4-bit/8-bit)
- Use appropriate batch sizes
- Enable caching with Redis
- Use faster storage (SSD)

**High memory usage:**
- Use gradient checkpointing
- Reduce sequence length
- Enable memory optimization flags
- Use smaller model variants

**Training instability:**
- Adjust learning rate
- Use learning rate scheduling
- Increase warmup steps
- Check data quality

## 📚 Additional Resources

### Documentation
- [Unsloth Documentation](https://docs.unsloth.ai/)
- [Transformers Documentation](https://huggingface.co/transformers/)
- [Docker Best Practices](https://docs.docker.com/develop/best-practices/)
- [Prometheus Monitoring](https://prometheus.io/docs/)

### Community
- [GitHub Issues](https://github.com/saptak/fine-tuning-small-llms/issues)
- [GitHub Discussions](https://github.com/saptak/fine-tuning-small-llms/discussions)
- [Hugging Face Community](https://huggingface.co/forums)
- [MLOps Community](https://mlops.community/)

### Blog Series
1. [Part 1: Setup and Environment](https://saptak.github.io/2025/07/25/fine-tuning-small-llms-part1-setup-environment/)
2. [Part 2: Data Preparation](https://saptak.github.io/2025/07/25/fine-tuning-small-llms-part2-data-preparation/)
3. [Part 3: Training with Unsloth](https://saptak.github.io/2025/07/25/fine-tuning-small-llms-part3-training/)
4. [Part 4: Evaluation and Testing](https://saptak.github.io/2025/07/25/fine-tuning-small-llms-part4-evaluation/)
5. [Part 5: Deployment](https://saptak.github.io/2025/07/25/fine-tuning-small-llms-part5-deployment/)
6. [Part 6: Production](https://saptak.github.io/2025/07/25/fine-tuning-small-llms-part6-production/)

---

**Need help?** Open an issue on [GitHub](https://github.com/saptak/fine-tuning-small-llms/issues) or check the [discussions](https://github.com/saptak/fine-tuning-small-llms/discussions) for community support.
