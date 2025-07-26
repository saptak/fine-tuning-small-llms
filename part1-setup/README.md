# Part 1: Setup and Environment

This directory contains all the setup scripts and configuration files needed to establish a complete Docker-based development environment for LLM fine-tuning.

## 🚀 Quick Setup

```bash
# Run the complete setup
./scripts/setup_environment.sh

# Or set up components individually
./scripts/install_docker.sh
./scripts/setup_cuda_support.sh
./scripts/install_python_deps.sh
```

## 📁 Directory Contents

- `src/` - Setup utilities and system detection scripts
- `configs/` - Docker configurations and environment templates
- `scripts/` - Installation and setup scripts
- `docs/` - Part 1 specific documentation

## 🔧 Features

- **Automated Docker Installation** with GPU support
- **CUDA Environment Setup** for NVIDIA GPUs
- **Python Environment** with all required dependencies
- **Development Tools** including Jupyter notebooks
- **Resource Monitoring** and system diagnostics

## 📋 Requirements Check

Run this to verify your system meets the requirements:

```bash
python src/system_check.py
```

## 🐳 Docker Environment

The setup includes:
- Base Python 3.10 environment
- CUDA support for GPU acceleration
- Jupyter notebook server
- Development tools and utilities

Start the development environment:

```bash
docker-compose -f configs/docker-compose.dev.yml up -d
```

## 📖 Related Blog Post

[Part 1: Setup and Environment](https://saptak.github.io/2025/07/25/fine-tuning-small-llms-part1-setup-environment/)
