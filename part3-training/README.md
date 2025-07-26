# Part 3: Fine-Tuning with Unsloth

This directory contains the complete training pipeline for fine-tuning models using Unsloth with LoRA adapters.

## 🎯 Features

- **Unsloth Integration**: 80% faster training with 80% less memory usage
- **LoRA Adapters**: Parameter-efficient fine-tuning
- **Multiple Model Support**: Llama, Mistral, Phi-3, CodeLlama, and more
- **Training Monitoring**: Weights & Biases integration
- **Automatic Checkpointing**: Resume training from interruptions
- **Model Quantization**: 4-bit and 8-bit training support

## 🚀 Quick Start

```bash
# Install training dependencies
pip install -r requirements.txt

# Train a SQL expert model
python src/fine_tune_model.py --config configs/sql_expert.yaml

# Monitor training with Jupyter
jupyter notebook notebooks/training_monitor.ipynb

# Convert to GGUF for deployment
python src/convert_to_gguf.py --model-path ./models/sql-expert-lora
```

## 📁 Directory Contents

- `src/` - Training scripts and utilities
- `configs/` - Training configuration files
- `notebooks/` - Interactive training notebooks
- `scripts/` - Automated training workflows
- `docs/` - Training guides and best practices

## 🔧 Key Components

### Training Script (`src/fine_tune_model.py`)
- Unsloth model loading and optimization
- LoRA adapter configuration
- Training loop with monitoring
- Automatic model saving

### Configuration System (`configs/`)
- YAML-based configuration files
- Model-specific hyperparameters
- Training pipeline settings
- Wandb integration config

### Model Conversion (`src/convert_to_gguf.py`)
- Merge LoRA adapters with base model
- Convert to GGUF format for Ollama
- Quantization options
- Model validation

## 📊 Training Monitoring

The training pipeline includes comprehensive monitoring:

- **Loss Tracking**: Training and validation loss curves
- **Learning Rate Scheduling**: Cosine annealing with warmup
- **Memory Usage**: GPU memory optimization tracking
- **Speed Metrics**: Tokens per second and time per epoch
- **Model Quality**: Validation metrics and sample outputs

## 🎨 Supported Models

- **Llama 3.1 8B**: General-purpose instruction following
- **Mistral 7B**: Technical and coding tasks
- **Phi-3 Mini**: Resource-constrained environments
- **CodeLlama 7B**: Pure code generation
- **Qwen2 7B**: Multilingual capabilities

## 📖 Related Blog Post

[Part 3: Fine-Tuning with Unsloth](https://saptak.github.io/2025/07/25/fine-tuning-small-llms-part3-training/)

## 🔗 Usage Examples

See the `notebooks/` directory for:
- Interactive training tutorials
- Hyperparameter tuning guides
- Model comparison notebooks
- Advanced training techniques
